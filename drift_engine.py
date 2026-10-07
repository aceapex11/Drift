"""
Drift engine for the existing project.

IMPORTANT:
This file is intentionally named `drift_engine.py` because that is the
filename used by the user's project.

Drift detection:
- Data drift: per-feature KS + PSI
- Relational drift: Page-Hinkley on residuals from a frozen old rule

ML adaptation:
1. Static
2. Full retraining
3. Incremental SGDRegressor
4. Continual + replay(100)
5. Continual + replay(500)

Ground-truth drift columns are NEVER used by live detection/adaptation.
"""

import copy
import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data_generator import FEATURES, MONITORED, TARGET

# Responsive drift-detector settings.  The model still evaluates each 600-hour
# stream batch, but drift is checked internally on smaller sub-windows so that
# a change can be detected earlier.  Thresholds are intentionally moderate:
# sensitivity is increased without making a single noisy feature an alarm.
KS_FLOOR = 0.10
PSI_THRESHOLD = 0.15
PH_PARAMS = dict(min_instances=50, delta=0.25, threshold=25.0, alpha=0.9999)

# Data drift is confirmed when at least two monitored features move in the
# same detector sub-window.  A 100-hour detector sub-window makes detection
# substantially faster than waiting for the full 600-hour model window.
DATA_DRIFT_MIN_FEATURES = 2
DATA_DRIFT_CONFIRM_WINDOWS = 2
DETECTOR_WINDOW = 100

RESID_CLIP = 4.0
RULE_BACK = 1.25
ADAPT_WINDOWS = 3

# Reference-notebook settings
ETA = 0.01
ALPHA = 1e-4
EPOCHS_PER_BATCH = 5
RETRAIN_MAX_ITER = 30
REPLAY_SIZES = (100, 500)
BUFFER_CAP = 2000


class PageHinkley:
    def __init__(self, min_instances=100, delta=0.05, threshold=60.0, alpha=0.9999):
        self.min_n = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.sum = 0.0
        self.stat = 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.sum = max(
            0.0,
            self.alpha * self.sum + (x - self.mean - self.delta),
        )
        self.stat = self.sum
        return self.n >= self.min_n and self.sum > self.threshold


def psi(ref, cur, bins=10):
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)

    q = np.quantile(ref, np.linspace(0, 1, bins + 1))
    q = np.unique(q)

    if len(q) < 2:
        return 0.0

    q[0] = -np.inf
    q[-1] = np.inf

    p0 = np.histogram(ref, q)[0] / len(ref) + 1e-6
    p1 = np.histogram(cur, q)[0] / len(cur) + 1e-6

    return float(np.sum((p1 - p0) * np.log(p1 / p0)))


def new_sgd(seed=0, max_iter=1000):
    return SGDRegressor(
        loss="squared_error",
        alpha=ALPHA,
        learning_rate="constant",
        eta0=ETA,
        max_iter=max_iter,
        tol=None,
        random_state=seed,
    )


def partial_epochs(model, Xb, yb, rng):
    for _ in range(EPOCHS_PER_BATCH):
        order = rng.permutation(len(yb))
        model.partial_fit(Xb[order], yb[order])


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    err = y_true - y_pred
    denom = np.sum((y_true - y_true.mean()) ** 2)

    r2 = np.nan if denom == 0 else 1.0 - np.sum(err ** 2) / denom

    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(r2),
    }


class Reservoir:
    def __init__(self, cap, n_features, rng):
        self.cap = int(cap)
        self.rng = rng
        self.n = 0
        self.X = np.empty((0, n_features), dtype=float)
        self.y = np.empty(0, dtype=float)

    def add(self, Xb, yb):
        for xi, yi in zip(Xb, yb):
            self.n += 1

            if len(self.y) < self.cap:
                self.X = np.vstack([self.X, xi])
                self.y = np.append(self.y, yi)
            else:
                j = self.rng.integers(0, self.n)
                if j < self.cap:
                    self.X[j] = xi
                    self.y[j] = yi

    def sample(self, k):
        if len(self.y) == 0:
            return self.X, self.y

        idx = self.rng.choice(
            len(self.y),
            min(int(k), len(self.y)),
            replace=False,
        )
        return self.X[idx], self.y[idx]


class Static:
    name = "Static"

    def __init__(self, base, seed=0):
        self.m = copy.deepcopy(base)

    def update(self, Xb, yb):
        return 0

    def predict(self, X):
        return self.m.predict(X)


class FullRetrain:
    name = "Full retraining"

    def __init__(self, base, seed, X_init, y_init):
        self.seed = seed
        self.Xall = X_init.copy()
        self.yall = y_init.copy()
        self.m = copy.deepcopy(base)

    def update(self, Xb, yb):
        self.Xall = np.vstack([self.Xall, Xb])
        self.yall = np.append(self.yall, yb)

        self.m = new_sgd(
            self.seed,
            RETRAIN_MAX_ITER,
        ).fit(self.Xall, self.yall)

        return len(self.yall)

    def predict(self, X):
        return self.m.predict(X)


class Incremental:
    name = "Incremental"

    def __init__(self, base, seed):
        self.m = copy.deepcopy(base)
        self.rng = np.random.default_rng(seed + 1)

    def update(self, Xb, yb):
        partial_epochs(self.m, Xb, yb, self.rng)
        return len(yb)

    def predict(self, X):
        return self.m.predict(X)


class Replay:
    def __init__(self, base, seed, k, X_init, y_init):
        self.k = int(k)
        self.name = f"Continual + replay({self.k})"

        self.m = copy.deepcopy(base)
        self.rng = np.random.default_rng(seed + 2 + self.k)

        self.mem = Reservoir(
            BUFFER_CAP,
            X_init.shape[1],
            self.rng,
        )
        self.mem.add(X_init, y_init)

    def update(self, Xb, yb):
        Xold, yold = self.mem.sample(self.k)

        if len(yold):
            Xt = np.vstack([Xb, Xold])
            yt = np.append(yb, yold)
        else:
            Xt, yt = Xb, yb

        partial_epochs(self.m, Xt, yt, self.rng)
        self.mem.add(Xb, yb)

        return len(yt)

    def predict(self, X):
        return self.m.predict(X)


STRATEGY_NAMES = [
    "Static",
    "Full retraining",
    "Incremental",
    "Continual + replay(100)",
    "Continual + replay(500)",
]


class SGDStreamEngine:
    """
    One streaming learner.

    Pipeline:
        predict -> detect -> diagnose -> adapt
    """

    def __init__(
        self,
        df,
        train_end,
        window,
        strategy="Incremental",
        seed=0,
    ):
        self.df = df
        self.window = int(window)
        self.train_end = int(train_end)
        self.seed = int(seed)

        Xraw = df[FEATURES].values.astype(float)
        yraw = df[TARGET].values.astype(float)

        self.scaler = StandardScaler().fit(Xraw[:train_end])
        self.X = self.scaler.transform(Xraw)

        self.y_mu = float(yraw[:train_end].mean())
        self.y_sd = float(yraw[:train_end].std()) or 1.0

        self.y = (yraw - self.y_mu) / self.y_sd
        self.yraw = yraw

        self.mon_idx = [FEATURES.index(f) for f in MONITORED]
        self.ref = self.X[:train_end]

        # Frozen rule for relational drift detection.
        self.rule = make_pipeline(
            PolynomialFeatures(2),
            Ridge(alpha=1.0),
        ).fit(self.ref, self.y[:train_end])

        rule_pred = self.rule.predict(self.ref)

        self.rule_sd = (
            float(np.sqrt(np.mean((self.y[:train_end] - rule_pred) ** 2)))
            or 1.0
        )

        r = np.abs(self.y[:train_end] - rule_pred)
        self.normal_res = float(np.mean(r / self.rule_sd)) or 1.0

        blocks = list(range(0, train_end - window + 1, window))

        self.ks_thr = {}
        for feature, j in zip(MONITORED, self.mon_idx):
            baseline_max = max(
                (
                    ks_2samp(
                        self.ref[b:b + window, j],
                        self.ref[:, j],
                    ).statistic
                    for b in blocks
                ),
                default=0.0,
            )

            self.ks_thr[feature] = max(
                KS_FLOOR,
                1.2 * baseline_max,
            )

        base = new_sgd(seed).fit(
            self.ref,
            self.y[:train_end],
        )

        X_init = self.ref.copy()
        y_init = self.y[:train_end].copy()

        if strategy == "Static":
            self.model = Static(base, seed)

        elif strategy == "Full retraining":
            self.model = FullRetrain(
                base,
                seed,
                X_init,
                y_init,
            )

        elif strategy == "Incremental":
            self.model = Incremental(base, seed)

        elif strategy == "Continual + replay(100)":
            self.model = Replay(
                base,
                seed,
                100,
                X_init,
                y_init,
            )

        elif strategy == "Continual + replay(500)":
            self.model = Replay(
                base,
                seed,
                500,
                X_init,
                y_init,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy

        ref_idx = np.arange(
            max(0, train_end - 2000),
            train_end,
        )

        self.reference_mae = (
            float(
                np.mean(
                    np.abs(
                        self.predict_raw(ref_idx)
                        - yraw[ref_idx]
                    )
                )
            )
            or 1.0
        )

        self.ph = PageHinkley(**PH_PARAMS)

        self.relational_on = False
        self.data_latched = False
        self.data_candidate_streak = 0
        self.alarm_episode_latched = False
        self.active = 0
        self.k = 0
        self.prev_diag = "none"
        self.update_count = 0

    def predict_raw(self, idx):
        return (
            self.model.predict(self.X[idx]) * self.y_sd
            + self.y_mu
        )

    def _detect(self, Xb, yb):
        """Detect data and relational drift with a fast sub-window scan.

        Xb/yb are normally one 600-hour model batch.  The detector scans that
        batch in smaller DETECTOR_WINDOW chunks.  This gives KS/PSI and
        Page-Hinkley a finer detection resolution while the model metrics still
        use the complete model window.

        KS + PSI monitor P(X). Page-Hinkley monitors P(Y|X) through residuals
        from the frozen reference rule.  The two families are complementary.
        """
        n = len(Xb)
        sub_starts = range(0, n, DETECTOR_WINDOW)

        all_ks = {f: 0.0 for f in MONITORED}
        all_psis = {f: 0.0 for f in MONITORED}
        moved_features = set()

        data_alarm = False
        rel_alarm = False
        raw_data = False
        relational_state = self.relational_on
        first_data_alarm = False
        first_rel_alarm = False
        first_alarm_offset = None
        max_ph = float(self.ph.stat)
        residual_parts = []

        # Scan the incoming model batch in small detector windows.
        for a in sub_starts:
            b = min(a + DETECTOR_WINDOW, n)
            Xs = Xb[a:b]
            ys = yb[a:b]

            ks_s = {
                f: float(
                    ks_2samp(
                        self.ref[:, j],
                        Xs[:, j],
                    ).statistic
                )
                for f, j in zip(MONITORED, self.mon_idx)
            }
            psi_s = {
                f: psi(
                    self.ref[:, j],
                    Xs[:, j],
                )
                for f, j in zip(MONITORED, self.mon_idx)
            }

            for f in MONITORED:
                all_ks[f] = max(all_ks[f], ks_s[f])
                all_psis[f] = max(all_psis[f], psi_s[f])

            moved_here = [
                f for f in MONITORED
                if (ks_s[f] > self.ks_thr[f])
                or (psi_s[f] > PSI_THRESHOLD)
            ]
            moved_features.update(moved_here)

            data_candidate = len(moved_here) >= DATA_DRIFT_MIN_FEATURES

            if data_candidate:
                self.data_candidate_streak += 1
            else:
                self.data_candidate_streak = 0

            raw_here = (
                self.data_candidate_streak
                >= DATA_DRIFT_CONFIRM_WINDOWS
            )

            if raw_here and not self.data_latched:
                data_alarm = True
                first_data_alarm = True
            raw_data = raw_here
            self.data_latched = raw_here

            # Frozen-rule residuals for relational drift.
            rule_pred = self.rule.predict(Xs)
            resid = np.minimum(
                np.abs(ys - rule_pred)
                / self.rule_sd
                / self.normal_res,
                RESID_CLIP,
            )
            residual_parts.append(resid)

            fired_here = False
            for value in resid:
                if self.ph.update(float(value)):
                    fired_here = True
                    self.ph.reset()

            max_ph = max(max_ph, float(self.ph.stat))

            if fired_here and not self.relational_on:
                rel_alarm = True
                first_rel_alarm = True
                self.relational_on = True

            if fired_here:
                relational_state = True
            elif (
                self.relational_on
                and float(resid.mean()) < RULE_BACK
            ):
                self.relational_on = False
                self.ph.reset()
                relational_state = False

            # Stop scanning only after both detector families have produced a
            # new alarm.  Otherwise keep scanning so recovery/state is correct.
            if first_alarm_offset is None and (first_data_alarm or first_rel_alarm):
                first_alarm_offset = a

        # `data_latched` represents the current data-drift state, not the
        # one-shot alarm.  This lets the detector recover when the distribution
        # returns to baseline.
        self.data_latched = bool(raw_data)

        if residual_parts:
            all_resid = np.concatenate(residual_parts)
            mean_resid = float(all_resid.mean())
        else:
            mean_resid = 0.0

        diagnosis = {
            (0, 0): "none",
            (1, 0): "data",
            (0, 1): "relational",
            (1, 1): "both",
        }[(int(raw_data), int(self.relational_on))]

        # One contiguous drift episode produces one NEW alarm.
        alarm_candidate = data_alarm or rel_alarm
        alarm = alarm_candidate and not self.alarm_episode_latched

        if alarm:
            self.alarm_episode_latched = True

        recovery = (
            self.prev_diag != "none"
            and diagnosis == "none"
        )

        if recovery:
            self.alarm_episode_latched = False
            self.data_candidate_streak = 0

        self.prev_diag = diagnosis

        return {
            "ks": all_ks,
            "psi": all_psis,
            "worst_ks": max(all_ks.values()) if all_ks else 0.0,
            "worst_psi": max(all_psis.values()) if all_psis else 0.0,
            "features_moved": sorted(moved_features),
            "data_alarm": bool(data_alarm),
            "relational_alarm": bool(rel_alarm),
            "alarm": bool(alarm),
            "recovery": bool(recovery),
            "diagnosis": diagnosis,
            "ph_stat": float(max_ph),
            "mean_resid": mean_resid,
            "detector_subwindows": int(np.ceil(n / DETECTOR_WINDOW)),
            "first_alarm_offset": first_alarm_offset,
        }


    def _adapt(self, Xb, yb, alarm, recovery):
        # A NEW drift alarm starts one adaptation cycle.
        # Recovery only marks the end of the drift episode; it does not
        # start another adaptation cycle.
        if alarm:
            self.active = ADAPT_WINDOWS

        adapted = False

        if self.active > 0:
            self.update_count += int(
                self.model.update(Xb, yb)
            )
            self.active -= 1
            adapted = True

        return adapted

    def step(self, s, e):
        idx = np.arange(s, e)

        Xb = self.X[idx]
        yb = self.y[idx]

        # Test first.
        pred = self.predict_raw(idx)
        metric = regression_metrics(
            self.yraw[idx],
            pred,
        )

        # Detect without ground truth. Detection scans smaller sub-windows
        # inside this model batch, so drift can be identified earlier than the
        # 600-hour model window. Prediction above was completed first, so the
        # reported metrics remain pre-adaptation and leakage-free.
        det = self._detect(Xb, yb)

        # Adapt after prediction and detection. The first alarm can trigger an
        # adaptation cycle using the current model batch.
        adapted = self._adapt(
            Xb,
            yb,
            det["alarm"],
            det["recovery"],
        )

        out = {
            "window": self.k,
            "start": int(s),
            "end": int(e),
            "strategy": self.strategy,
            **metric,
            "mae_x_normal": float(
                metric["mae"] / self.reference_mae
            ),
            "diagnosis": det["diagnosis"],
            "alarm": det["alarm"],
            "recovery": det["recovery"],
            "data_alarm": det["data_alarm"],
            "relational_alarm": det["relational_alarm"],
            "adapted": bool(adapted),
            "ph_stat": det["ph_stat"],
            "mean_resid": det["mean_resid"],
            "detector_subwindows": det["detector_subwindows"],
            "first_alarm_offset": det["first_alarm_offset"],
            "ks": det["ks"],
            "psi": det["psi"],
            "worst_ks": det["worst_ks"],
            "worst_psi": det["worst_psi"],
            "features_moved": det["features_moved"],
            "updates_seen": int(self.update_count),
        }

        self.k += 1
        return out


def make_windows(n, train_end, window):
    return [
        (s, min(s + window, n))
        for s in range(train_end, n, window)
    ]


class StrategyRunner:
    """
    Run all five strategies on the same stream.

    The detector is maintained separately inside each strategy,
    while all strategies receive the same data windows.
    """

    def __init__(
        self,
        df,
        train_end,
        window,
        seed=0,
        strategies=None,
    ):
        self.df = df
        self.train_end = int(train_end)
        self.window = int(window)
        self.seed = int(seed)

        self.strategies = (
            strategies
            if strategies is not None
            else STRATEGY_NAMES
        )

        self.engines = {
            name: SGDStreamEngine(
                df,
                train_end,
                window,
                name,
                seed,
            )
            for name in self.strategies
        }

        self.windows = make_windows(
            len(df),
            train_end,
            window,
        )

    def step(self, s, e):
        return {
            name: engine.step(s, e)
            for name, engine in self.engines.items()
        }

    def run(self):
        records = []

        for s, e in self.windows:
            batch = self.step(s, e)

            for strategy, row in batch.items():
                records.append(row)

        return records


class DualRunner:
    """
    Compatibility runner for the Streamlit live dashboard.

    IMPORTANT:
    The live model is SGD incremental learning.
    There is no TransferGBR and no transfer-learning dependency.
    """

    def __init__(
        self,
        df,
        train_end,
        window,
        seed=0,
    ):
        self.live = SGDStreamEngine(
            df,
            train_end,
            window,
            "Incremental",
            seed,
        )

        self.static = SGDStreamEngine(
            df,
            train_end,
            window,
            "Static",
            seed,
        )

        self.reference_mae = self.live.reference_mae
        self.ks_thr = self.live.ks_thr

    def step(self, s, e):
        out = self.live.step(s, e)
        static = self.static.step(s, e)

        out["mae_static"] = static["mae"]
        out["rmse_static"] = static["rmse"]
        out["r2_static"] = static["r2"]

        return out
