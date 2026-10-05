"""
Drift engine = classic statistical drift detection + incremental/continual ML.

DATA drift       -> per-feature KS test + PSI
RELATIONAL drift -> Page-Hinkley on residuals from a frozen old-rule (Poly-2 Ridge)
PREDICTOR        -> SGDRegressor

Adaptation strategies inspired by the supplied reference notebook:
1. Static             : no update
2. Full retraining    : refit from scratch on all data seen so far
3. Incremental        : partial_fit() on the newest batch
4. Continual replay   : partial_fit() on newest batch + replay memory

Ground-truth columns are never used by the live detector or adaptation.
"""

import copy
import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data_generator import FEATURES, MONITORED, TARGET

KS_FLOOR = 0.15
PSI_THRESHOLD = 0.25
PH_PARAMS = dict(min_instances=100, delta=1.0, threshold=100.0, alpha=0.9999)
RESID_CLIP = 4.0
RULE_BACK = 1.25
ADAPT_WINDOWS = 3

# Reference-notebook SGD settings
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
        self.sum = max(0.0, self.alpha * self.sum + (x - self.mean - self.delta))
        self.stat = self.sum
        return self.n >= self.min_n and self.sum > self.threshold


def psi(ref, cur, bins=10):
    q = np.quantile(ref, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    p0 = np.histogram(ref, q)[0] / len(ref) + 1e-6
    p1 = np.histogram(cur, q)[0] / len(cur) + 1e-6
    return float(np.sum((p1 - p0) * np.log(p1 / p0)))


def new_sgd(seed=0, max_iter=1000):
    """Same core SGDRegressor setup as the supplied incremental-learning notebook."""
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
    """Several shuffled partial_fit passes over one streaming update batch."""
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
    """Fixed-size random memory using reservoir sampling."""
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
        idx = self.rng.choice(len(self.y), min(int(k), len(self.y)), replace=False)
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

    def __init__(self, base, seed, X_init, y_init, n_features):
        self.seed = seed
        self.Xall = X_init.copy()
        self.yall = y_init.copy()
        self.m = copy.deepcopy(base)
        self.n_features = n_features

    def update(self, Xb, yb):
        self.Xall = np.vstack([self.Xall, Xb])
        self.yall = np.append(self.yall, yb)
        self.m = new_sgd(self.seed, RETRAIN_MAX_ITER).fit(self.Xall, self.yall)
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
    def __init__(self, base, seed, k, X_init, y_init, n_features):
        self.k = int(k)
        self.name = f"Continual + replay({self.k})"
        self.m = copy.deepcopy(base)
        self.rng = np.random.default_rng(seed + 2 + self.k)
        self.mem = Reservoir(BUFFER_CAP, n_features, self.rng)
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
    """predict -> detect -> diagnose -> adapt for one SGD strategy."""

    def __init__(self, df, train_end, window, strategy="Incremental", seed=0):
        self.df = df
        self.window = window
        self.train_end = train_end
        self.seed = seed

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

        # Frozen detector rule. It is NOT updated during adaptation.
        self.rule = make_pipeline(
            PolynomialFeatures(2),
            Ridge(alpha=1.0),
        ).fit(self.ref, self.y[:train_end])
        rule_pred = self.rule.predict(self.ref)
        self.rule_sd = float(np.sqrt(np.mean((self.y[:train_end] - rule_pred) ** 2))) or 1.0
        r = np.abs(self.y[:train_end] - rule_pred)
        self.normal_res = float(np.mean(r / self.rule_sd)) or 1.0

        blocks = range(0, train_end - window + 1, window)
        self.ks_thr = {
            f: max(
                KS_FLOOR,
                1.2 * max(
                    ks_2samp(self.ref[b:b + window, j], self.ref[:, j]).statistic
                    for b in blocks
                ),
            )
            for f, j in zip(MONITORED, self.mon_idx)
        }

        base = new_sgd(seed).fit(self.ref, self.y[:train_end])
        self.base_model = base
        X_init = self.ref.copy()
        y_init = self.y[:train_end].copy()
        n_features = self.X.shape[1]

        if strategy == "Static":
            self.model = Static(base, seed)
        elif strategy == "Full retraining":
            self.model = FullRetrain(base, seed, X_init, y_init, n_features)
        elif strategy == "Incremental":
            self.model = Incremental(base, seed)
        elif strategy == "Continual + replay(100)":
            self.model = Replay(base, seed, 100, X_init, y_init, n_features)
        elif strategy == "Continual + replay(500)":
            self.model = Replay(base, seed, 500, X_init, y_init, n_features)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy
        self.reference_mae = float(
            np.mean(
                np.abs(
                    self.predict_raw(np.arange(max(0, train_end - 2000), train_end))
                    - yraw[max(0, train_end - 2000):train_end]
                )
            )
        ) or 1.0

        self.ph = PageHinkley(**PH_PARAMS)
        self.relational_on = False
        self.data_latched = False
        self.active = 0
        self.k = 0
        self.prev_diag = "none"
        self.rng = np.random.default_rng(seed + 11)
        self.src_pool = np.arange(train_end)
        self.update_count = 0

    def predict_raw(self, idx):
        return self.model.predict(self.X[idx]) * self.y_sd + self.y_mu

    def _detect(self, Xb, yb):
        ks = {
            f: float(ks_2samp(self.X[:, j][:self.train_end], Xb[:, j]).statistic)
            for f, j in zip(MONITORED, self.mon_idx)
        }
        # Correct reference for KS/PSI is the fixed source history.
        ks = {
            f: float(ks_2samp(self.ref[:, j], Xb[:, j]).statistic)
            for f, j in zip(MONITORED, self.mon_idx)
        }
        psis = {
            f: psi(self.ref[:, j], Xb[:, j])
            for f, j in zip(MONITORED, self.mon_idx)
        }

        raw_data = any(v > self.ks_thr[f] for f, v in ks.items())
        data_alarm = raw_data and not self.data_latched
        self.data_latched = raw_data

        rule_pred = self.rule.predict(Xb)
        resid = np.minimum(
            np.abs(yb - rule_pred) / self.rule_sd / self.normal_res,
            RESID_CLIP,
        )
        fired = False
        for v in resid:
            if self.ph.update(float(v)):
                fired = True
                self.ph.reset()

        rel_alarm = fired and not self.relational_on
        if fired:
            self.relational_on = True
        elif self.relational_on and resid.mean() < RULE_BACK:
            self.relational_on = False
            self.ph.reset()

        diagnosis = {
            (0, 0): "none",
            (1, 0): "data",
            (0, 1): "relational",
            (1, 1): "both",
        }[(int(raw_data), int(self.relational_on))]
        alarm = data_alarm or rel_alarm
        recovery = self.prev_diag != "none" and diagnosis == "none"
        self.prev_diag = diagnosis

        return {
            "ks": ks,
            "psi": psis,
            "worst_ks": max(ks.values()) if ks else 0.0,
            "worst_psi": max(psis.values()) if psis else 0.0,
            "features_moved": [f for f, v in ks.items() if v > self.ks_thr[f]],
            "data_alarm": bool(data_alarm),
            "relational_alarm": bool(rel_alarm),
            "alarm": bool(alarm),
            "recovery": bool(recovery),
            "diagnosis": diagnosis,
            "ph_stat": float(self.ph.stat),
            "mean_resid": float(resid.mean()),
        }

    def _adapt(self, Xb, yb, diagnosis, alarm, recovery):
        if alarm or recovery:
            self.active = ADAPT_WINDOWS

        adapted = False
        if self.active > 0:
            # The notebook's replay strategies do their own memory handling.
            # The detector supplies only the trigger; the learner receives the
            # current batch. This avoids leaking diagnosis/ground truth into the learner.
            self.update_count += int(self.model.update(Xb, yb))
            self.active -= 1
            adapted = True
        return adapted

    def step(self, s, e):
        idx = np.arange(s, e)
        Xb, yb = self.X[idx], self.y[idx]

        # 1) Predict before learning from this batch (test-then-train).
        pred = self.predict_raw(idx)
        metric = regression_metrics(self.yraw[idx], pred)

        # 2) Detect and diagnose without ground-truth drift labels.
        det = self._detect(Xb, yb)

        # 3) Adapt only after prediction + detection.
        adapted = self._adapt(
            Xb, yb,
            det["diagnosis"],
            det["alarm"],
            det["recovery"],
        )

        out = {
            "window": self.k,
            "start": int(s),
            "end": int(e),
            "strategy": self.strategy,
            **metric,
            "mae_x_normal": float(metric["mae"] / self.reference_mae),
            "diagnosis": det["diagnosis"],
            "alarm": det["alarm"],
            "recovery": det["recovery"],
            "data_alarm": det["data_alarm"],
            "relational_alarm": det["relational_alarm"],
            "adapted": bool(adapted),
            "ph_stat": det["ph_stat"],
            "mean_resid": det["mean_resid"],
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
    return [(s, min(s + window, n)) for s in range(train_end, n, window)]


class StrategyRunner:
    """Runs all five notebook-style strategies against the same streaming detector."""
    def __init__(self, df, train_end, window, seed=0, strategies=None):
        self.df = df
        self.train_end = train_end
        self.window = window
        self.seed = seed
        self.strategies = strategies or STRATEGY_NAMES
        self.engines = {
            name: SGDStreamEngine(df, train_end, window, name, seed)
            for name in self.strategies
        }
        self.windows = make_windows(len(df), train_end, window)

    def step(self, s, e):
        rows = {}
        for name, engine in self.engines.items():
            rows[name] = engine.step(s, e)
        return rows

    def run(self):
        records = []
        for s, e in self.windows:
            batch = self.step(s, e)
            for name, row in batch.items():
                records.append(row)
        return records


# Backwards-friendly dashboard runner: incremental live model vs static twin.
class DualRunner:
    def __init__(self, df, train_end, window, seed=0):
        self.live = SGDStreamEngine(df, train_end, window, "Incremental", seed)
        self.static = SGDStreamEngine(df, train_end, window, "Static", seed)
        self.reference_mae = self.live.reference_mae
        self.ks_thr = self.live.ks_thr

    def step(self, s, e):
        out = self.live.step(s, e)
        static = self.static.step(s, e)
        out["mae_static"] = static["mae"]
        out["rmse_static"] = static["rmse"]
        out["r2_static"] = static["r2"]
        return out
