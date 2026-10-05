"""
Drift engine = detectors (classic ML / statistics, no deep learning) + transfer-learning predictor.

DATA drift        -> per-feature KS test (effect size) on each incoming batch, PSI as severity
RELATIONAL drift  -> Page-Hinkley on the residual of a frozen "old rule" (poly-2 Ridge fitted on normal data)
PREDICTOR         -> Transfer learning with gradient boosting: pre-train on source history, FREEZE the
                     existing trees, and on a drift alarm fine-tune by boosting extra trees on the new data.
"""
import copy
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data_generator import FEATURES, MONITORED, TARGET

KS_FLOOR = 0.15              # per-feature alarm level = max(floor, 1.2 x the largest KS seen in normal training blocks)
PSI_THRESHOLD = 0.25         # industry rule of thumb: > 0.25 = major shift
PH_PARAMS = dict(min_instances=100, delta=1.0, threshold=100.0, alpha=0.9999)   # residual is in units of "normal error" (1.0 = normal)
RESID_CLIP = 4.0             # winsorise residuals so a 10-row sensor glitch cannot trigger the alarm
RULE_BACK = 1.25             # relational evidence clears once old-rule error < 1.25 x normal
ADAPT_WINDOWS = 3            # after an alarm, keep fine-tuning for this many windows
FT_TREES = 80                # extra trees added per fine-tune step
FT_LR = 0.15
SRC_REPLAY = 2500             # source-domain rows replayed when only the sensor distribution moved


class PageHinkley:
    """Page-Hinkley test for an upward shift in the mean (same maths as river.drift.PageHinkley)."""
    def __init__(self, min_instances=100, delta=0.05, threshold=60.0, alpha=0.9999):
        self.min_n, self.delta, self.threshold, self.alpha = min_instances, delta, threshold, alpha
        self.reset()

    def reset(self):
        self.n, self.mean, self.sum, self.stat = 0, 0.0, 0.0, 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.sum = max(0.0, self.alpha * self.sum + (x - self.mean - self.delta))
        self.stat = self.sum
        return self.n >= self.min_n and self.sum > self.threshold


def psi(ref, cur, bins=10):
    q = np.quantile(ref, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -np.inf, np.inf
    p0 = np.histogram(ref, q)[0] / len(ref) + 1e-6
    p1 = np.histogram(cur, q)[0] / len(cur) + 1e-6
    return float(np.sum((p1 - p0) * np.log(p1 / p0)))


def new_gbr(seed=0, iters=250):
    # classic GBM (not the histogram variant): it keeps raw split thresholds, so warm-start fine-tuning on NEW data is valid
    return GradientBoostingRegressor(n_estimators=iters, learning_rate=0.08, max_depth=4, min_samples_leaf=20,
                                     subsample=0.8, warm_start=True, random_state=seed)


class TransferGBR:
    """Pre-train once on the source domain; later fine-tune by continuing the boosting on new data."""
    def __init__(self, seed=0):
        self.seed, self.model = seed, None
        self.n_source_trees = 0

    def pretrain(self, X, y):
        self.model = new_gbr(self.seed).fit(X, y)
        self.n_source_trees = self.model.n_estimators_
        return self

    def fine_tune(self, X, y, w=None):
        self.model.set_params(n_estimators=self.model.n_estimators_ + FT_TREES, learning_rate=FT_LR)
        self.model.fit(X, y, sample_weight=w)                          # warm start: old trees are frozen
        return self

    def predict(self, X): return self.model.predict(X)

    @property
    def n_trees(self): return self.model.n_estimators_


class RetrainGBR:
    """Benchmark: throw everything away and fit a brand-new model on the recent buffer."""
    def __init__(self, seed=0): self.seed, self.model = seed, None
    def pretrain(self, X, y): self.model = new_gbr(self.seed).fit(X, y); return self
    def fine_tune(self, X, y, w=None): self.model = new_gbr(self.seed).fit(X, y, sample_weight=w); return self
    def predict(self, X): return self.model.predict(X)


class StaticGBR(TransferGBR):
    def fine_tune(self, X, y, w=None): return self


class StreamEngine:
    """predict -> detect -> diagnose -> adapt, one window at a time. Never reads the ground-truth columns."""
    def __init__(self, df, train_end, window, model_cls=TransferGBR, seed=0, pretrained=None):
        self.df, self.window = df, window
        Xraw = df[FEATURES].values.astype(float); yraw = df[TARGET].values.astype(float)
        self.scaler = StandardScaler().fit(Xraw[:train_end])
        self.X = self.scaler.transform(Xraw)
        self.y_mu, self.y_sd = yraw[:train_end].mean(), yraw[:train_end].std()
        self.y = (yraw - self.y_mu) / self.y_sd
        self.yraw = yraw
        self.train_end = train_end
        self.mon_idx = [FEATURES.index(f) for f in MONITORED]
        self.ref = self.X[:train_end]

        # frozen "old rule" used only for relational-drift detection
        self.rule = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0)).fit(self.ref, self.y[:train_end])
        r = np.abs(self.y[:train_end] - self.rule.predict(self.ref))
        self.rule_sd = float(np.sqrt(np.mean((self.y[:train_end] - self.rule.predict(self.ref)) ** 2)))
        self.normal_res = float(np.mean(r / self.rule_sd))

        # empirical null for the KS alarm: how far does a normal 600-row block stray from the whole normal history?
        blocks = range(0, train_end - window + 1, window)
        self.ks_thr = {f: max(KS_FLOOR, 1.2 * max(ks_2samp(self.ref[b:b + window, j], self.ref[:, j]).statistic for b in blocks))
                       for f, j in zip(MONITORED, self.mon_idx)}

        if pretrained is None:
            self.model = model_cls(seed).pretrain(self.ref, self.y[:train_end])
        else:                                                                   # reuse an already pre-trained source model
            self.model = copy.deepcopy(pretrained); self.model.__class__ = model_cls
        self.reference_mae = float(np.mean(np.abs(self.predict_raw(np.arange(train_end - 2000, train_end)) - yraw[train_end - 2000:train_end])))
        self.ph = PageHinkley(**PH_PARAMS)
        self.relational_on = self.data_latched = False
        self.active, self.k = 0, 0
        self.buf_X, self.buf_y = [], []
        self.prev_diag = "none"
        self.rng = np.random.default_rng(seed + 11)
        self.src_pool = np.arange(train_end)

    def predict_raw(self, idx): return self.model.predict(self.X[idx]) * self.y_sd + self.y_mu

    def step(self, s, e):
        idx = np.arange(s, e)
        Xb, yb = self.X[idx], self.y[idx]
        pred = self.predict_raw(idx)                                            # 1) predict (test-then-train)
        mae = float(np.mean(np.abs(pred - self.yraw[idx])))

        ks = {f: float(ks_2samp(self.X[idx, j], self.ref[:, j]).statistic) for f, j in zip(MONITORED, self.mon_idx)}   # 2) detect
        psis = {f: psi(self.ref[:, j], self.X[idx, j]) for f, j in zip(MONITORED, self.mon_idx)}
        raw_data = any(v > self.ks_thr[f] for f, v in ks.items())
        data_alarm = raw_data and not self.data_latched
        self.data_latched = raw_data

        resid = np.minimum(np.abs(yb - self.rule.predict(Xb)) / self.rule_sd / self.normal_res, RESID_CLIP)
        fired = False
        for v in resid:
            if self.ph.update(float(v)):
                fired = True; self.ph.reset()
        rel_alarm = fired and not self.relational_on
        if fired: self.relational_on = True
        elif self.relational_on and resid.mean() < RULE_BACK:
            self.relational_on = False; self.ph.reset()

        diagnosis = {(0, 0): "none", (1, 0): "data", (0, 1): "relational", (1, 1): "both"}[(int(raw_data), int(self.relational_on))]
        alarm = data_alarm or rel_alarm
        recovery = self.prev_diag != "none" and diagnosis == "none"             # drift just ENDED -> model may now be stale
        self.prev_diag = diagnosis

        adapted = False                                                         # 3) adapt
        if alarm or recovery: self.active = ADAPT_WINDOWS
        if self.active > 0:
            self.buf_X.append(Xb); self.buf_y.append(yb)
            self.buf_X, self.buf_y = self.buf_X[-2:], self.buf_y[-2:]           # last 2 windows only
            Xr, yr = np.vstack(self.buf_X), np.concatenate(self.buf_y)
            wr = np.linspace(0.5, 1.0, len(yr))                                 # newest rows count most
            if diagnosis in ("relational", "both"):                             # old relation is invalid -> learn from recent data only
                Xt, yt, wt = Xr, yr, wr
            else:                                                               # data-only drift / recovery -> keep old knowledge: replay source rows
                src = self.rng.choice(self.src_pool, SRC_REPLAY, replace=False)
                Xt, yt, wt = np.vstack([Xr, self.ref[src]]), np.concatenate([yr, self.y[src]]), np.concatenate([wr, np.full(SRC_REPLAY, 0.5)])
            self.model.fine_tune(Xt, yt, wt)
            self.active -= 1; adapted = True
        out = dict(window=self.k, start=int(s), end=int(e), mae=mae, mae_x_normal=mae / self.reference_mae,
                   diagnosis=diagnosis, alarm=bool(alarm), recovery=bool(recovery), data_alarm=bool(data_alarm), relational_alarm=bool(rel_alarm),
                   adapted=adapted, ph_stat=float(self.ph.stat), mean_resid=float(resid.mean()),
                   ks=ks, psi=psis, worst_ks=max(ks.values()), worst_psi=max(psis.values()),
                   features_moved=[f for f, v in ks.items() if v > self.ks_thr[f]])
        self.k += 1
        return out


def make_windows(n, train_end, window):
    return [(s, min(s + window, n)) for s in range(train_end, n, window)]


class DualRunner:
    """Runs the live (transfer-learning) model next to a frozen 'static' twin so a dashboard can show the improvement."""
    def __init__(self, df, train_end, window, pretrained=None, seed=0):
        self.live = StreamEngine(df, train_end, window, TransferGBR, seed, pretrained)
        self.static = StreamEngine(df, train_end, window, StaticGBR, seed, self.live.model)
        self.reference_mae = self.live.reference_mae
        self.ks_thr = self.live.ks_thr

    def step(self, s, e):
        out = self.live.step(s, e); out["mae_static"] = self.static.step(s, e)["mae"]
        out["n_trees"] = self.live.model.n_trees
        return out
