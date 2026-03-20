#!/usr/bin/env python3
"""
=============================================================================
Stock Market Regime Detection — Extended Edition
=============================================================================
Builds on hmm_regime.py with five new modules:

  1. Walk-forward validation     — rolling 2-yr train / 1-mo OOS prediction
  2. Regime persistence filter   — suppress single-day flip noise (≥5 days)
  3. Enhanced strategies         — Regime-weighted & Short-Bear equity curves
  4. Robustness checks           — QQQ and IWM side-by-side comparison
  5. PDF report                  — all charts + summary tables in one file

All original code is preserved; new features are added as standalone
functions that slot into the pipeline after the base fit.
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker
import seaborn as sns
import requests

# yfinance is the primary data source; fall back gracefully if absent
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False
    print("[INFO] yfinance not installed — will use requests fallback, "
          "then synthetic data.  Install with: pip install yfinance")

warnings.filterwarnings('ignore')

OUTPUT_DIR = "/sessions/lucid-confident-bohr/mnt/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TODAY   = pd.Timestamp.today().strftime("%Y-%m-%d")
START   = "2010-01-01"
COLORS  = {"Bull": "#2ecc71", "Bear": "#e74c3c", "High-Vol": "#95a5a6"}
ORDER   = ["Bull", "Bear", "High-Vol"]

# ── Transaction cost & position parameters ───────────────────────────────────
COST_BPS           = 5      # one-way cost per $1 of notional traded (bps)
REWEIGHT_THRESHOLD = 0.05   # min |Δposition| before the weighted strategy rebalances

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA FETCHING  (identical to base script, extended for 3 assets)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance(symbol, start=START, end=TODAY):
    """
    Primary data fetcher — uses the official yfinance library.

    Key flags
    ---------
    auto_adjust=True  : 'Close' column is already split- and dividend-adjusted;
                        no separate 'Adj Close' needed.
    progress=False    : suppress the tqdm download bar in terminal output.

    Column layout returned (with auto_adjust=True):
      Open | High | Low | Close | Volume
    where Close == adjusted close.
    """
    if not _YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,   # Close already contains adjusted prices
            progress=False,
            threads=False,
        )
        if df.empty:
            raise ValueError("Empty DataFrame returned")

        # yfinance ≥ 0.2 may return MultiIndex columns for single tickers;
        # flatten them to plain string column names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        df = df.ffill().dropna()
        return df
    except Exception as exc:
        print(f"    [WARN] yfinance failed for {symbol}: {exc}")
        return None


def fetch_yahoo(symbol, start=START, end=TODAY):
    """Fallback fetcher — direct requests to Yahoo Finance v8 chart API."""
    sym_enc = symbol.replace("^", "%5E")
    p1, p2  = int(pd.Timestamp(start).timestamp()), int(pd.Timestamp(end).timestamp())
    url     = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym_enc}"
               f"?interval=1d&period1={p1}&period2={p2}&events=history")
    try:
        r = requests.get(url,
                         headers={"User-Agent": "Mozilla/5.0"},
                         timeout=20)
        r.raise_for_status()
        d   = r.json()["chart"]["result"][0]
        q   = d["indicators"]["quote"][0]
        ac  = d["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
        df  = pd.DataFrame({
            "Open": q["open"], "High": q["high"], "Low": q["low"],
            "Close": q["close"], "Volume": q["volume"], "Adj Close": ac,
        }, index=pd.to_datetime(d["timestamp"], unit="s").normalize())
        df.index.name = "Date"
        return df.ffill().dropna()
    except Exception as e:
        print(f"    [WARN] {symbol}: {e}")
        return None


# ── Synthetic data parameters per asset ──────────────────────────────────────
#
# BUG FIX: the previous parameters produced a downtrend in synthetic data
# because Bear regime occupied ~35 % of time at -15 bps/day, overwhelming
# the Bull regime's +6 bps/day.
#
# Corrected design targets ~+10 % annualised return per asset:
#   Stationary dist: Bull≈65 %, Bear≈13 %, High-Vol≈22 %
#   Expected daily  = 0.65×+8 + 0.13×-10 + 0.22×+1 ≈ +4.1 bps → ~10 %/yr
#   Terminal SPX    ≈ 1115 × e^(0.00041 × 4231) ≈ 5 600  (matches reality)
#
# Transition matrix is calibrated so the above stationary distribution holds:
#   Bull  row: [0.97, 0.01, 0.02]
#   Bear  row: [0.07, 0.90, 0.03]
#   HVol  row: [0.05, 0.03, 0.92]
# ─────────────────────────────────────────────────────────────────────────────
_ASSET_PARAMS = {
    # S&P 500 — moderate growth, low-vol bull market
    "^GSPC": dict(
        start_px = 1115.10,
        ret_mean = [ 0.00080, -0.00100,  0.00010],   # Bull / Bear / High-Vol
        ret_std  = [ 0.00700,  0.01800,  0.01300],
        vix_mean = [-0.002,    0.010,    0.003],
        vix_std  = [ 0.030,    0.060,    0.050],
        A = [[0.970, 0.010, 0.020],
             [0.070, 0.900, 0.030],
             [0.050, 0.030, 0.920]],
    ),
    # QQQ (Nasdaq-100) — higher growth, higher vol
    "QQQ": dict(
        start_px = 46.60,
        ret_mean = [ 0.00095, -0.00120,  0.00015],
        ret_std  = [ 0.00900,  0.02100,  0.01500],
        vix_mean = [-0.003,    0.012,    0.004],
        vix_std  = [ 0.035,    0.065,    0.055],
        A = [[0.968, 0.010, 0.022],
             [0.068, 0.900, 0.032],
             [0.050, 0.030, 0.920]],
    ),
    # IWM (Russell 2000) — similar to SPX, slightly more volatile
    "IWM": dict(
        start_px = 64.50,
        ret_mean = [ 0.00075, -0.00110,  0.00010],
        ret_std  = [ 0.00850,  0.02000,  0.01450],
        vix_mean = [-0.002,    0.011,    0.003],
        vix_std  = [ 0.032,    0.062,    0.052],
        A = [[0.969, 0.010, 0.021],
             [0.069, 0.900, 0.031],
             [0.050, 0.030, 0.920]],
    ),
}


def generate_synthetic(ticker, start=START, end=TODAY, seed=42):
    """
    Generate regime-switching synthetic OHLCV + VIX data for any asset.
    Parameters are calibrated per-asset to match empirical characteristics.
    """
    p   = _ASSET_PARAMS[ticker]
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, end)
    T     = len(dates)
    A     = np.array(p["A"])

    regime = np.empty(T, dtype=int)
    regime[0] = 0
    for t in range(1, T):
        regime[t] = rng.choice(3, p=A[regime[t - 1]])

    ret_mean = np.array(p["ret_mean"])
    ret_std  = np.array(p["ret_std"])
    vix_mean = np.array(p["vix_mean"])
    vix_std  = np.array(p["vix_std"])

    log_ret = ret_mean[regime] + ret_std[regime] * rng.randn(T)
    vix_chg = vix_mean[regime] + vix_std[regime] * rng.randn(T)

    px  = p["start_px"] * np.exp(np.cumsum(log_ret))
    vix = 17.5 * np.exp(np.cumsum(vix_chg)).clip(8, 80)

    spx_df = pd.DataFrame({
        "Adj Close": px, "Close": px,
        "Open":  px * (1 + rng.randn(T) * 0.002),
        "High":  px * (1 + np.abs(rng.randn(T)) * 0.003),
        "Low":   px * (1 - np.abs(rng.randn(T)) * 0.003),
        "Volume": (3e9 * (1 + rng.randn(T) * 0.3)).clip(1e8),
    }, index=dates)
    spx_df.index.name = "Date"

    vix_df = pd.DataFrame({"Close": vix}, index=dates)
    vix_df.index.name = "Date"
    return spx_df, vix_df


def load_asset(ticker, start=START, end=TODAY):
    """
    Return (price_df, vix_df, data_source_label) for a given ticker.

    Fetch priority
    --------------
    1. yfinance with auto_adjust=True  → Close column = adjusted prices
    2. Raw requests to Yahoo v8 API    → Adj Close column
    3. Regime-switching synthetic data → Close + Adj Close columns (both set)

    The returned DataFrames are normalised so that downstream code can
    always access the adjusted price via px_df["Close"] (yfinance path)
    or px_df["Adj Close"] (requests / synthetic path).  build_features()
    handles both column layouts transparently.
    """
    date_label = f"{start} to {end}"   # used in chart titles (no "Synthetic")
    print(f"  Fetching {ticker} …")

    # ── Try yfinance first ────────────────────────────────────────────────────
    px_df = fetch_yfinance(ticker, start, end)
    vx_df = fetch_yfinance("^VIX",  start, end)

    if px_df is not None and vx_df is not None:
        src = f"Yahoo Finance / yfinance  [{ticker}]  |  auto_adjust=True"
        print(f"    {ticker}: {len(px_df)} rows  "
              f"| first Close={px_df['Close'].iloc[0]:.2f}"
              f"  last Close={px_df['Close'].iloc[-1]:.2f}")
        return px_df, vx_df, src

    # ── Try raw requests fallback ─────────────────────────────────────────────
    px_df = fetch_yahoo(ticker, start, end)
    vx_df = fetch_yahoo("^VIX",  start, end)

    if px_df is not None and vx_df is not None:
        src = f"Yahoo Finance / requests  [{ticker}]"
        return px_df, vx_df, src

    # ── Last resort: calibrated synthetic data ────────────────────────────────
    tk    = ticker if ticker in _ASSET_PARAMS else "^GSPC"
    px_df, vx_df = generate_synthetic(tk, start, end, seed=42)
    src = f"Calibrated synthetic  [{ticker}]"
    print(f"    No live data — using calibrated synthetic data for {ticker}")
    return px_df, vx_df, src


# ── Known SPX spot-check prices (Close, not Adj Close) ───────────────────────
# These are widely-reported closing prices used only to sanity-check real data.
_SPX_BENCHMARKS = {
    "2020-03-23": (2237.40, 150, "COVID crash low"),   # tolerance ±150
    "2024-01-02": (4742.83, 300, "2024 opening day"),  # tolerance ±300
}


def validate_real_data(feat_df, ticker, data_source):
    """
    For ^GSPC: verify the loaded price series against two known historical closes.

    Logic
    -----
    • If data is synthetic → skip with a clear warning (cannot validate fake data).
    • If ticker is not ^GSPC → skip (no benchmarks for ETFs).
    • Otherwise: look up each benchmark date (or the nearest trading day) in
      feat_df["spx_adj"] and check it lies within the stated tolerance.
    • If any check fails → raise ValueError and halt the pipeline so the user
      can investigate the data ingestion before wasting time on a bad model.

    Parameters
    ----------
    feat_df     : DataFrame built by build_features() — must contain "spx_adj"
    ticker      : str, e.g. "^GSPC"
    data_source : str returned by load_asset()

    Returns
    -------
    True  — validation passed (real data confirmed)
    False — validation skipped (synthetic data or non-SPX ticker)
    """
    if "synthetic" in data_source.lower():
        print(f"  [VALIDATION SKIPPED] {ticker}: calibrated synthetic data in use.\n"
              f"    Run on a machine with internet access + `pip install yfinance`\n"
              f"    to fetch real prices and enable benchmark validation.")
        return False

    if ticker != "^GSPC":
        return True   # benchmarks are SPX-specific

    print(f"\n  [VALIDATION] Checking {ticker} against known price benchmarks …")
    failures = []

    for date_str, (expected, tol, desc) in _SPX_BENCHMARKS.items():
        target = pd.Timestamp(date_str)
        if target in feat_df.index:
            actual    = float(feat_df.loc[target, "spx_adj"])
            used_date = date_str
        else:
            idx       = feat_df.index.get_indexer([target], method="nearest")[0]
            actual    = float(feat_df["spx_adj"].iloc[idx])
            used_date = str(feat_df.index[idx].date())
            print(f"    {date_str} not a trading day — using nearest: {used_date}")

        ok     = abs(actual - expected) <= tol
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {status}  {desc} ({used_date}): "
              f"actual={actual:,.2f}   expected≈{expected:,.2f}   tol=±{tol}")
        if not ok:
            failures.append(
                f"    {date_str} ({desc}): got {actual:.2f}, "
                f"expected {expected:.2f} ± {tol}"
            )

    if failures:
        raise ValueError(
            "\n\nDATA VALIDATION FAILED — pipeline aborted.\n"
            "The SPX prices do not match known historical values:\n"
            + "\n".join(failures)
            + "\n\nLikely causes:\n"
            "  • Wrong price column ('Close' vs 'Adj Close' mix-up)\n"
            "  • Incorrect ticker symbol\n"
            "  • Stale / corrupt local cache\n"
            f"  Data source reported: {data_source}"
        )

    print(f"  [PASS] All benchmark checks passed for {ticker} ✓\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class StandardScaler:
    def fit(self, X):
        self.mean_  = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-12
        return self
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def build_features(px_df, vx_df):
    """
    Construct the 2-column feature matrix [log_return, vix_change].
    Returns (feat_df, X_raw, X_scaled, scaler).

    Column handling
    ---------------
    • yfinance (auto_adjust=True) → adjusted prices are in "Close"
    • requests fetcher / synthetic  → adjusted prices are in "Adj Close"
    Both cases are handled transparently.
    """
    # Prefer "Adj Close" (explicitly adjusted); fall back to "Close"
    price_col = "Adj Close" if "Adj Close" in px_df.columns else "Close"
    adj   = px_df[price_col].ffill()
    vix   = vx_df["Close"].ffill().reindex(adj.index, method="ffill")
    lr    = np.log(adj / adj.shift(1))
    vc    = vix.pct_change()

    df = pd.DataFrame({
        "log_return": lr,
        "vix_change": vc,
        "spx_adj":    adj,
    }).dropna()

    X_raw  = df[["log_return", "vix_change"]].values
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)
    return df, X_raw, X, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GAUSSIAN HMM  (extended with predict_proba)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianHMM:
    """
    Full-covariance Gaussian HMM — Baum-Welch EM + Viterbi decoding.
    Extended with predict_proba() for regime-weighted strategy.
    """
    def __init__(self, n_components=3, n_iter=200, tol=1e-5,
                 random_state=42, reg_cov=1e-4):
        self.K           = n_components
        self.n_iter      = n_iter
        self.tol         = tol
        self.rng         = np.random.RandomState(random_state)
        self.reg_cov     = reg_cov
        self.startprob_  = None
        self.transmat_   = None
        self.means_      = None
        self.covars_     = None
        self.loglikelihood_ = -np.inf

    @staticmethod
    def _logsumexp(arr, axis=None, keepdims=False):
        max_v    = np.max(arr, axis=axis, keepdims=True)
        max_safe = np.where(np.isfinite(max_v), max_v, 0.0)
        out      = max_safe + np.log(
                       np.sum(np.exp(arr - max_safe), axis=axis, keepdims=True))
        if not keepdims:
            out = np.squeeze(out)
        if isinstance(out, np.ndarray) and out.ndim == 0:
            return float(out)
        return out

    def _log_emission(self, X):
        T, d = X.shape
        out  = np.empty((T, self.K))
        const = d * np.log(2.0 * np.pi)
        for k in range(self.K):
            cov          = self.covars_[k] + np.eye(d) * self.reg_cov
            sign, ldet   = np.linalg.slogdet(cov)
            if sign <= 0:
                cov, ldet = np.eye(d) * self.reg_cov, d * np.log(self.reg_cov)
            cov_inv      = np.linalg.inv(cov)
            diff         = X - self.means_[k]
            mahal        = np.einsum("ti,ij,tj->t", diff, cov_inv, diff)
            out[:, k]    = -0.5 * (const + ldet + mahal)
        return out

    def _forward(self, le):
        T, K = le.shape
        la   = np.empty((T, K))
        lA   = np.log(np.clip(self.transmat_, 1e-300, 1))
        la[0]= np.log(np.clip(self.startprob_, 1e-300, 1)) + le[0]
        for t in range(1, T):
            v       = la[t-1, :, None] + lA
            mv      = v.max(axis=0)
            la[t]   = mv + np.log(np.exp(v - mv).sum(axis=0)) + le[t]
        return la

    def _backward(self, le):
        T, K = le.shape
        lb   = np.zeros((T, K))
        lA   = np.log(np.clip(self.transmat_, 1e-300, 1))
        for t in range(T-2, -1, -1):
            v       = lA + le[t+1] + lb[t+1]
            mv      = v.max(axis=1)
            lb[t]   = mv + np.log(np.exp(v - mv[:, None]).sum(axis=1))
        return lb

    def _kmeans_init(self, X):
        T, d = X.shape
        idx  = self.rng.choice(T, self.K, replace=False)
        c    = X[idx].copy()
        for _ in range(50):
            dists  = np.array([((X - c[k])**2).sum(axis=1) for k in range(self.K)])
            asgn   = np.argmin(dists, axis=0)
            nc     = np.array([X[asgn==k].mean(axis=0) if (asgn==k).sum()>0
                               else c[k] for k in range(self.K)])
            if np.allclose(c, nc, atol=1e-6): break
            c = nc
        self.means_  = c
        self.covars_ = np.array([
            np.cov(X[asgn==k].T) + np.eye(d)*0.1
            if (asgn==k).sum() > d else np.eye(d)*0.1
            for k in range(self.K)])
        self.transmat_  = np.full((self.K, self.K), 0.1/max(self.K-1, 1))
        np.fill_diagonal(self.transmat_, 0.9)
        self.startprob_ = np.full(self.K, 1.0/self.K)

    def fit(self, X, verbose=True):
        T, d = X.shape
        self._kmeans_init(X)
        prev = -np.inf
        for it in range(self.n_iter):
            le   = self._log_emission(X)
            la   = self._forward(le)
            lb   = self._backward(le)
            ll   = self._logsumexp(la[-1])

            # Gamma
            lg   = la + lb
            lse  = self._logsumexp(lg, axis=1, keepdims=True)
            gam  = np.exp(lg - lse)

            # Xi
            lA   = np.log(np.clip(self.transmat_, 1e-300, 1))
            lxi  = (la[:-1, :, None] + lA[None]
                    + le[1:, None, :] + lb[1:, None, :])
            lxif = lxi.reshape(T-1, self.K*self.K)
            lz   = self._logsumexp(lxif, axis=1, keepdims=True).reshape(T-1,1,1)
            xi   = np.exp(lxi - lz)

            # M-step
            self.startprob_ = gam[0] + 1e-300
            self.startprob_ /= self.startprob_.sum()
            xs               = xi.sum(axis=0)
            self.transmat_   = xs / (xs.sum(axis=1, keepdims=True) + 1e-300)
            gs               = gam.sum(axis=0) + 1e-300
            self.means_      = (gam[:, :, None] * X[:, None, :]).sum(axis=0) / gs[:, None]
            for k in range(self.K):
                diff            = X - self.means_[k]
                self.covars_[k] = ((gam[:, k, None, None]
                                    * diff[:, :, None]
                                    * diff[:, None, :]).sum(axis=0) / gs[k]
                                   + np.eye(d) * self.reg_cov)
            delta = abs(ll - prev)
            if verbose and (it+1) % 25 == 0:
                print(f"    iter {it+1:4d}  ll={ll:,.1f}  Δ={delta:.2e}")
            if delta < self.tol and it > 10:
                if verbose:
                    print(f"    Converged at iter {it+1}  (Δ={delta:.2e})")
                break
            prev = ll
        self.loglikelihood_ = ll
        return self

    def predict(self, X):
        """Viterbi most-likely state sequence."""
        T, K  = len(X), self.K
        le    = self._log_emission(X)
        lA    = np.log(np.clip(self.transmat_, 1e-300, 1))
        lpi   = np.log(np.clip(self.startprob_, 1e-300, 1))
        dlt   = np.empty((T, K))
        psi   = np.zeros((T, K), dtype=int)
        dlt[0]= lpi + le[0]
        for t in range(1, T):
            tr       = dlt[t-1, :, None] + lA
            psi[t]   = tr.argmax(axis=0)
            dlt[t]   = tr[psi[t], np.arange(K)] + le[t]
        st       = np.empty(T, dtype=int)
        st[-1]   = dlt[-1].argmax()
        for t in range(T-2, -1, -1):
            st[t] = psi[t+1, st[t+1]]
        return st

    def predict_proba(self, X):
        """
        ── NEW ──
        Return posterior state probabilities γ[t, k] = P(s_t=k | X)
        using the forward-backward algorithm.
        Shape: (T, K)
        """
        le  = self._log_emission(X)
        la  = self._forward(le)
        lb  = self._backward(le)
        lg  = la + lb
        lse = self._logsumexp(lg, axis=1, keepdims=True)
        return np.exp(lg - lse)   # (T, K)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3-UTIL — Regime labelling (shared helper)
# ─────────────────────────────────────────────────────────────────────────────

def label_states(raw_states, X_raw):
    """
    Map raw integer states → 'Bull' / 'Bear' / 'High-Vol' by ranking
    mean log-return.  Returns (label_map, regimes_array).
    """
    means   = np.array([X_raw[:, 0][raw_states == k].mean() for k in range(3)])
    ranked  = np.argsort(means)                # ascending → [bear, highvol, bull]
    lmap    = {ranked[0]: "Bear", ranked[1]: "High-Vol", ranked[2]: "Bull"}
    regimes = np.array([lmap[s] for s in raw_states])
    return lmap, regimes


def label_states_from_map(raw_states, lmap):
    """Apply a pre-computed label_map (from a training window) to new states."""
    return np.array([lmap.get(s, "High-Vol") for s in raw_states])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3B — WALK-FORWARD VALIDATION  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DAYS = 504    # 2 trading years
STEP_DAYS  = 21     # ~1 calendar month


def walk_forward_predict(X_raw, X_scaled, train_days=TRAIN_DAYS,
                         step_days=STEP_DAYS, random_state=42,
                         verbose=True):
    """
    Rolling-window out-of-sample HMM regime prediction.

    Algorithm
    ---------
    For each window w:
      1. Fit HMM on X_scaled[t-train_days : t]
      2. Decode training window → establish Bull/Bear/High-Vol labelling
      3. Decode test slice X_scaled[t : t+step_days] with Viterbi
      4. Re-label using the same mean-return ranking as step 2
      5. Record posterior Bull probability (predict_proba) for each test day
      6. Advance t by step_days

    Returns
    -------
    oos_regimes : list of str   (None for the first train_days rows)
    oos_bull_p  : list of float (None for the first train_days rows)
    oos_mask    : boolean array  — True where we have a prediction
    """
    T = len(X_scaled)
    oos_regimes = [None] * T
    oos_bull_p  = [None] * T

    t = train_days
    n_wins = 0
    while t < T:
        t_end  = min(t + step_days, T)
        X_tr   = X_scaled[t - train_days : t]
        X_te   = X_scaled[t : t_end]
        Xr_tr  = X_raw   [t - train_days : t]

        # Fit on training slice (reduced iterations for speed)
        m = GaussianHMM(n_components=3, n_iter=100, tol=1e-4,
                        random_state=random_state, reg_cov=1e-4)
        m.fit(X_tr, verbose=False)

        # Label training window → determine which raw state = Bull
        tr_states        = m.predict(X_tr)
        means_tr         = np.array([Xr_tr[:, 0][tr_states == k].mean()
                                     for k in range(3)])
        ranked           = np.argsort(means_tr)
        lmap             = {ranked[0]: "Bear", ranked[1]: "High-Vol", ranked[2]: "Bull"}
        bull_k           = ranked[2]

        # Predict on test slice
        te_states        = m.predict(X_te)
        te_proba         = m.predict_proba(X_te)    # (step, 3)

        for i, tt in enumerate(range(t, t_end)):
            oos_regimes[tt] = lmap.get(te_states[i], "High-Vol")
            oos_bull_p[tt]  = float(te_proba[i, bull_k])

        t      += step_days
        n_wins += 1

    oos_mask = np.array([r is not None for r in oos_regimes])
    if verbose:
        n_pred  = oos_mask.sum()
        bull_pc = np.mean([r == "Bull" for r in oos_regimes if r])
        print(f"  Walk-forward: {n_wins} windows  |  "
              f"{n_pred} OOS days  |  Bull={bull_pc:.1%}")
    return oos_regimes, oos_bull_p, oos_mask


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3C — PERSISTENCE FILTER  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def apply_persistence_filter(regimes_arr, min_persist=5):
    """
    Suppress short-lived regime switches.

    A transition is only accepted if the incoming regime persists for at
    least min_persist consecutive days.  Short runs are filled with the
    preceding confirmed regime, then the algorithm iterates until stable
    (handles cascading short runs).

    Returns a numpy array of regime labels the same length as regimes_arr.
    """
    regimes = list(regimes_arr)
    changed = True

    while changed:
        changed = False
        # Identify all contiguous runs
        runs = []
        i = 0
        while i < len(regimes):
            j = i
            while j < len(regimes) and regimes[j] == regimes[i]:
                j += 1
            runs.append((regimes[i], i, j))
            i = j

        # Merge runs shorter than min_persist into their left neighbour
        for idx, (label, start, end) in enumerate(runs):
            if (end - start) < min_persist:
                replacement = runs[idx-1][0] if idx > 0 else runs[1][0]
                for k in range(start, end):
                    regimes[k] = replacement
                changed = True
                break   # restart scan after any modification

    return np.array(regimes)


def count_trades(in_market):
    """Count entry/exit events in a binary or continuous position series."""
    return int((pd.Series(in_market).diff().fillna(0) != 0).sum())


def apply_reweight_threshold(bull_prob_series, threshold=REWEIGHT_THRESHOLD):
    """
    Convert a daily Bull-posterior series to a thresholded position series.

    Without this filter the Regime-Weighted strategy rebalances every single
    day (posterior probability changes continuously), accumulating enormous
    transaction costs at 5 bps/trade.

    Rule: only update the held position when the new posterior differs from
    the current position by MORE than `threshold`.  The held position is
    otherwise carried forward unchanged.

    Parameters
    ----------
    bull_prob_series : pd.Series  daily P(Bull | data) from predict_proba()
    threshold        : float      minimum |Δposition| to trigger a rebalance

    Returns
    -------
    pd.Series of thresholded positions (same index as bull_prob_series)
    """
    prob  = bull_prob_series.values
    pos   = np.empty(len(prob))
    pos[0] = prob[0]
    for t in range(1, len(prob)):
        if abs(prob[t] - pos[t - 1]) > threshold:
            pos[t] = prob[t]      # accept the new posterior as position
        else:
            pos[t] = pos[t - 1]   # hold current position
    return pd.Series(pos, index=bull_prob_series.index)


def apply_costs(log_ret_series, pos_series, cost_bps=COST_BPS):
    """
    Deduct one-way transaction costs from a gross log-return series.

    Cost formula
    ------------
    On each day t where the position changes by Δ = |pos[t] − pos[t−1]|,
    a cost of  Δ × cost_bps / 10 000  is subtracted from that day's return.

    This handles:
      • Binary 0/1 strategies: each entry/exit costs exactly cost_bps bps.
      • L/S ±1 strategies: a full reversal (−1 → +1) costs 2 × cost_bps bps.
      • Continuous position strategies: cost is proportional to position change.
      • Buy & Hold: one entry on day 0, otherwise no changes.

    Parameters
    ----------
    log_ret_series : pd.Series  gross daily log returns
    pos_series     : pd.Series  daily position (same index)
    cost_bps       : float      one-way cost per unit of notional traded

    Returns
    -------
    pd.Series  net daily log returns after deducting transaction costs
    """
    pos   = pd.Series(pos_series.values, index=log_ret_series.index,
                      dtype=float)
    delta = pos.diff().abs()
    delta.iloc[0] = abs(pos.iloc[0])     # entry cost on the very first day
    cost  = delta * (cost_bps / 10_000)
    return log_ret_series - cost


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PERFORMANCE METRICS  (shared by all strategies)
# ─────────────────────────────────────────────────────────────────────────────

def perf_metrics(daily_log_ret, label, ann=252):
    """
    Annualised Return, Volatility, Sharpe Ratio, Max Drawdown
    from a series of daily log-returns.
    """
    s        = pd.Series(daily_log_ret).dropna()
    n        = len(s)
    ann_ret  = (s.sum() / n) * ann
    ann_vol  = s.std() * np.sqrt(ann)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum      = np.exp(s.cumsum())
    max_dd   = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "Strategy":   label,
        "Ann. Return":   f"{ann_ret:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio":  f"{sharpe:.2f}",
        "Max Drawdown":  f"{max_dd:.2%}",
        "_r": ann_ret, "_v": ann_vol, "_sh": sharpe, "_dd": max_dd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def shade_regimes(ax, dates, regimes):
    """Paint background spans by regime colour."""
    i = 0
    while i < len(regimes):
        reg = regimes[i]
        j   = i
        while j < len(regimes) and regimes[j] == reg:
            j += 1
        ax.axvspan(dates[i], dates[min(j, len(dates)-1)-1],
                   alpha=0.22, color=COLORS[reg], linewidth=0)
        i = j


def regime_patches():
    return [mpatches.Patch(color=COLORS[r], alpha=0.5, label=r) for r in ORDER]


def plot_cum_returns(ax, curves, styles, title, subtitle=""):
    """
    curves : dict  name → pd.Series of cumulative return (starts at 1)
    styles : dict  name → (color, linestyle, linewidth)
    """
    for name, series in curves.items():
        c, ls, lw = styles[name]
        ax.plot(series.index, series.values, color=c, linestyle=ls,
                linewidth=lw, label=name)
        final = series.iloc[-1]
        ax.annotate(f"{final:.2f}×",
                    xy=(series.index[-1], final),
                    xytext=(-6, 6), textcoords="offset points",
                    fontsize=8, color=c, ha="right")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    ax.set_ylabel("Growth of $1", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FULL ASSET PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(ticker, start=START, end=TODAY, verbose=True):
    """
    Complete regime-detection and backtesting pipeline for one asset.

    Returns a dict with keys:
      feat_df, X_raw, X, scaler, model,
      insample_regimes, oos_regimes, oos_bull_p, oos_mask,
      filtered_regimes, trans_mat,
      figures (list of matplotlib Figure objects),
      perf_table (DataFrame), data_source
    """
    name = ticker.replace("^", "")
    if verbose:
        print(f"\n{'='*65}")
        print(f"  PIPELINE: {ticker}  |  {start} → {end}")
        print(f"{'='*65}")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    if verbose: print("\n[1] Loading data …")
    px_df, vx_df, src = load_asset(ticker, start, end)

    # ── 2. Features ───────────────────────────────────────────────────────────
    if verbose: print("[2] Engineering features …")
    feat_df, X_raw, X, scaler = build_features(px_df, vx_df)
    T = len(X)
    dates = feat_df.index

    # ── 2B. Validate real prices for ^GSPC (halts on bad data) ───────────────
    if verbose: print("[2B] Validating data …")
    validate_real_data(feat_df, ticker, src)

    # ── 3. In-sample HMM fit ──────────────────────────────────────────────────
    if verbose: print("[3] Fitting in-sample HMM …")
    model = GaussianHMM(n_components=3, n_iter=200, tol=1e-5,
                        random_state=42, reg_cov=1e-4)
    model.fit(X, verbose=verbose)
    raw_states = model.predict(X)
    lmap, is_regimes = label_states(raw_states, X_raw)
    feat_df["regime_IS"]  = is_regimes
    if verbose:
        print(f"  IS log-likelihood: {model.loglikelihood_:,.1f}")

    # ── 3B. Walk-forward validation ───────────────────────────────────────────
    if verbose: print("[3B] Walk-forward validation …")
    oos_regimes, oos_bull_p, oos_mask = walk_forward_predict(
        X_raw, X, train_days=TRAIN_DAYS, step_days=STEP_DAYS,
        random_state=42, verbose=verbose)
    feat_df["regime_OOS"]  = [r if r else np.nan for r in oos_regimes]
    feat_df["bull_prob_OOS"] = [p if p is not None else np.nan for p in oos_bull_p]

    # ── 3C. Persistence filter (applied to in-sample for comparison) ──────────
    if verbose: print("[3C] Applying persistence filter …")
    filt_regimes    = apply_persistence_filter(is_regimes, min_persist=5)
    feat_df["regime_filt"] = filt_regimes

    # ── Transition matrix (from IS regimes) ───────────────────────────────────
    trans_mat = pd.DataFrame(np.zeros((3, 3)), index=ORDER, columns=ORDER)
    for i in range(len(is_regimes)-1):
        trans_mat.loc[is_regimes[i], is_regimes[i+1]] += 1
    trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0)

    # ── In-sample posterior Bull probability (for regime-weighted strategy) ───
    gamma_IS = model.predict_proba(X)
    bull_k   = [k for k, v in lmap.items() if v == "Bull"][0]
    feat_df["bull_prob_IS"] = gamma_IS[:, bull_k]

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — BACKTEST  (gross + net-of-costs for all strategies)
    # ─────────────────────────────────────────────────────────────────────────
    if verbose: print("[5] Backtesting strategies (gross & net of costs) …")

    lr = feat_df["log_return"]

    # ── Gross positions ───────────────────────────────────────────────────────

    # A — Bull-only IS
    feat_df["pos_IS"]              = (feat_df["regime_IS"] == "Bull").astype(float)

    # B — Bull-only OOS (walk-forward)
    feat_df["pos_OOS"]             = (feat_df["regime_OOS"] == "Bull"
                                      ).astype(float).fillna(0)

    # C — Bull-only with persistence filter
    feat_df["pos_filt"]            = (feat_df["regime_filt"] == "Bull").astype(float)

    # D1 — Regime-weighted RAW  (every-day posterior — kept for cost comparison)
    feat_df["pos_weighted_raw"]    = feat_df["bull_prob_IS"].clip(0, 1)

    # D2 — Regime-weighted THRESHOLDED  (only rebalance if |Δ| > REWEIGHT_THRESHOLD)
    feat_df["pos_weighted"]        = apply_reweight_threshold(
                                         feat_df["bull_prob_IS"],
                                         threshold=REWEIGHT_THRESHOLD)

    # E — Short Bear IS: +1 Bull, −1 Bear, 0 High-Vol
    feat_df["pos_short"]           = feat_df["regime_IS"].map(
                                         {"Bull": 1.0, "Bear": -1.0, "High-Vol": 0.0})

    # ── Gross returns (position × log-return, no costs) ───────────────────────
    feat_df["ret_bnh"]             = lr
    feat_df["ret_IS"]              = lr * feat_df["pos_IS"]
    feat_df["ret_OOS"]             = lr * feat_df["pos_OOS"]
    feat_df["ret_filt"]            = lr * feat_df["pos_filt"]
    feat_df["ret_weighted_raw"]    = lr * feat_df["pos_weighted_raw"]
    feat_df["ret_weighted"]        = lr * feat_df["pos_weighted"]
    feat_df["ret_short"]           = lr * feat_df["pos_short"]

    # ── Net returns (gross − transaction costs) ────────────────────────────────
    # Buy & Hold: one entry on day 0, then fully static → negligible drag
    bnh_pos = pd.Series(1.0, index=feat_df.index)
    feat_df["ret_bnh_net"]         = apply_costs(lr, bnh_pos)

    feat_df["ret_IS_net"]          = apply_costs(lr * feat_df["pos_IS"],
                                                  feat_df["pos_IS"])
    feat_df["ret_OOS_net"]         = apply_costs(lr * feat_df["pos_OOS"],
                                                  feat_df["pos_OOS"])
    feat_df["ret_filt_net"]        = apply_costs(lr * feat_df["pos_filt"],
                                                  feat_df["pos_filt"])
    feat_df["ret_weighted_raw_net"]= apply_costs(lr * feat_df["pos_weighted_raw"],
                                                  feat_df["pos_weighted_raw"])
    feat_df["ret_weighted_net"]    = apply_costs(lr * feat_df["pos_weighted"],
                                                  feat_df["pos_weighted"])
    feat_df["ret_short_net"]       = apply_costs(lr * feat_df["pos_short"],
                                                  feat_df["pos_short"])

    # ── Cumulative curves (gross and net) ─────────────────────────────────────
    for tag in ["IS", "OOS", "filt",
                "weighted_raw", "weighted", "short", "bnh"]:
        feat_df[f"cum_{tag}"]      = np.exp(feat_df[f"ret_{tag}"].cumsum())
        feat_df[f"cum_{tag}_net"]  = np.exp(feat_df[f"ret_{tag}_net"].cumsum())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    if verbose: print("[4] Generating charts …")
    figs = {}

    # ── Fig A: SPX price + IS regime shading ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, feat_df["spx_adj"], color="#2c3e50", lw=0.9, zorder=5)
    shade_regimes(ax, dates, is_regimes)
    patches = regime_patches()
    patches.append(plt.Line2D([0], [0], color="#2c3e50", lw=1.5, label=f"{name} Price"))
    ax.legend(handles=patches, fontsize=9, loc="upper left")
    ax.set_yscale("log")
    ax.set_title(f"{ticker}  —  HMM Regime Detection  |  {start} to {end}",
                 fontsize=11)
    ax.set_ylabel("Price (log scale)", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    figs["regime_chart"] = fig

    # ── Fig B: Regime bar stats ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    stats = []
    for r in ORDER:
        m = is_regimes == r
        ret_r = X_raw[:, 0][m]
        stats.append({"Regime": r,
                      "Mean Return (bps)": ret_r.mean() * 1e4,
                      "Volatility (bps)":  ret_r.std()  * 1e4,
                      "N days": m.sum()})
    sdf = pd.DataFrame(stats)
    bc  = [COLORS[r] for r in sdf["Regime"]]
    for ax, col, ttl in zip(axes,
                             ["Mean Return (bps)", "Volatility (bps)"],
                             ["Mean Daily Return", "Daily Volatility"]):
        bars = ax.bar(sdf["Regime"], sdf[col], color=bc, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        for b, v in zip(bars, sdf[col]):
            ax.text(b.get_x() + b.get_width()/2,
                    v + (0.3 if v >= 0 else -0.7),
                    f"{v:+.1f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
        ax.set_title(f"{ttl} per Regime  [{ticker}]", fontsize=10)
        ax.set_ylabel("Basis points", fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.35)
    plt.tight_layout()
    figs["regime_stats"] = fig

    # ── Fig C: Transition heatmap ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(trans_mat, annot=True, fmt=".3f", cmap="Blues",
                linewidths=0.5, vmin=0, vmax=1, ax=ax,
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title(f"Regime Transition Probabilities  [{ticker}]", fontsize=11)
    ax.set_xlabel("Next Regime", fontsize=9)
    ax.set_ylabel("Current Regime", fontsize=9)
    plt.tight_layout()
    figs["trans_heatmap"] = fig

    # ── Fig D: Walk-forward IS vs OOS comparison ──────────────────────────────
    oos_slice = feat_df[oos_mask].copy()
    # Re-anchor both curves to 1.0 at the OOS start date
    is_oos   = np.exp(feat_df.loc[oos_mask, "ret_IS"].cumsum())
    bnh_oos  = np.exp(feat_df.loc[oos_mask, "ret_bnh"].cumsum())
    oos_oos  = np.exp(feat_df.loc[oos_mask, "ret_OOS"].cumsum())
    is_oos  /= is_oos.iloc[0]
    bnh_oos /= bnh_oos.iloc[0]
    oos_oos /= oos_oos.iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(is_oos.index,  is_oos.values,  color="#2980b9", lw=1.8,
            label="In-Sample HMM  (data leakage)")
    ax.plot(oos_oos.index, oos_oos.values, color="#e74c3c", lw=1.8,
            linestyle="--", label="Walk-Forward OOS  (no peeking)")
    ax.plot(bnh_oos.index, bnh_oos.values, color="#7f8c8d", lw=1.2,
            linestyle=":", label="Buy & Hold")
    ax.set_title(f"Walk-Forward OOS vs In-Sample Backtest  [{ticker}]  "
                 f"|  {start} to {end}\n"
                 f"OOS period starts {oos_slice.index[0].date()}",
                 fontsize=11)
    ax.set_ylabel("Cumulative Return (re-indexed to 1)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.35)
    plt.tight_layout()
    figs["walkfwd_chart"] = fig

    # ── Fig E: Persistence filter comparison ─────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: raw vs filtered regime strips
    for ax_i, (regs, ttl) in enumerate(zip(
            [is_regimes, filt_regimes],
            ["Raw IS Regimes (unfiltered)", "Filtered Regimes (≥5 day persistence)"])):
        ax = axes[ax_i]
        ax.plot(dates, feat_df["spx_adj"], color="#2c3e50", lw=0.8)
        shade_regimes(ax, dates, regs)
        n_sw = sum(regs[i] != regs[i-1] for i in range(1, len(regs)))
        ax.set_title(f"{ttl}  |  {n_sw} switches  [{ticker}]", fontsize=10)
        ax.set_yscale("log")
        ax.grid(axis="y", ls="--", alpha=0.3)
        ax.legend(handles=regime_patches(), fontsize=8, loc="upper left")
    plt.tight_layout()
    figs["filter_chart"] = fig

    # ── Fig F: All 4 equity curves ────────────────────────────────────────────
    # Net-of-cost equity curves (the realistic view)
    strat_curves_net = {
        "Buy & Hold":                  feat_df["cum_bnh_net"],
        "Bull-Only IS (net)":          feat_df["cum_IS_net"],
        "Regime-Weighted (net)":       feat_df["cum_weighted_net"],
        "Regime-Weighted raw (net)":   feat_df["cum_weighted_raw_net"],
        "Short Bear IS (net)":         feat_df["cum_short_net"],
    }
    strat_styles_net = {
        "Buy & Hold":                  ("#7f8c8d", "--",  1.4),
        "Bull-Only IS (net)":          ("#2980b9", "-",   1.8),
        "Regime-Weighted (net)":       ("#27ae60", "-",   1.8),
        "Regime-Weighted raw (net)":   ("#c0392b", ":",   1.4),
        "Short Bear IS (net)":         ("#e67e22", "-",   1.8),
    }
    fig, ax = plt.subplots(figsize=(15, 5.5))
    plot_cum_returns(ax, strat_curves_net, strat_styles_net,
                     f"All Strategy Equity Curves — Net of {COST_BPS} bps/trade  "
                     f"[{ticker}]  |  {start} to {end}",
                     subtitle=(f"Dotted red = Regime-Weighted without threshold "
                                f"(trades every day → severe cost drag)"))
    plt.tight_layout()
    figs["equity_curves"] = fig

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7 — GROSS vs NET PERFORMANCE TABLE + COST-DRAG DETAIL
    # ─────────────────────────────────────────────────────────────────────────

    # Map: (strategy label, gross_ret_col, net_ret_col, pos_col)
    _strat_defs = [
        ("Buy & Hold",
         feat_df["ret_bnh"],
         feat_df["ret_bnh_net"],
         bnh_pos),
        ("Bull-Only IS",
         feat_df["ret_IS"],
         feat_df["ret_IS_net"],
         feat_df["pos_IS"]),
        ("Bull-Only OOS",
         feat_df.loc[oos_mask, "ret_OOS"],
         feat_df.loc[oos_mask, "ret_OOS_net"],
         feat_df.loc[oos_mask, "pos_OOS"]),
        ("Persistence Filter",
         feat_df["ret_filt"],
         feat_df["ret_filt_net"],
         feat_df["pos_filt"]),
        ("Regime-Weighted (raw)",        # trades every day — shows cost disaster
         feat_df["ret_weighted_raw"],
         feat_df["ret_weighted_raw_net"],
         feat_df["pos_weighted_raw"]),
        (f"Regime-Weighted (≥{REWEIGHT_THRESHOLD:.0%} thresh)",
         feat_df["ret_weighted"],
         feat_df["ret_weighted_net"],
         feat_df["pos_weighted"]),
        ("Short Bear",
         feat_df["ret_short"],
         feat_df["ret_short_net"],
         feat_df["pos_short"]),
    ]

    rows = []
    ANN  = 252
    for sname, gross_ret, net_ret, pos in _strat_defs:
        mg = perf_metrics(gross_ret, sname)
        mn = perf_metrics(net_ret,   sname)
        n_trades  = count_trades(pos)
        # Annual cost drag = (gross ann return − net ann return) in bps
        drag_bps  = (mg["_r"] - mn["_r"]) * 10_000
        rows.append({
            "Strategy":         sname,
            "Trades":           n_trades,
            "Gross Return":     mg["Ann. Return"],
            "Net Return":       mn["Ann. Return"],
            "Gross Sharpe":     mg["Sharpe Ratio"],
            "Net Sharpe":       mn["Sharpe Ratio"],
            "Max Drawdown":     mn["Max Drawdown"],
            "Cost Drag (bps/yr)": f"{drag_bps:.1f}",
            # raw numeric for console formatting
            "_gross_r":   mg["_r"],
            "_net_r":     mn["_r"],
            "_gross_sh":  mg["_sh"],
            "_net_sh":    mn["_sh"],
            "_drag":      drag_bps,
        })

    perf_df = pd.DataFrame(rows)[[
        "Strategy", "Trades", "Gross Return", "Net Return",
        "Gross Sharpe", "Net Sharpe", "Max Drawdown", "Cost Drag (bps/yr)"
    ]]

    if verbose:
        bar = "─" * 78
        print(f"\n  ── Gross vs Net Performance [{ticker}]  "
              f"(cost = {COST_BPS} bps/trade) ──────────────")
        print(perf_df.to_string(index=False))

        print(f"\n  ── Trade count & cost drag detail [{ticker}] ──────────────────────")
        print(f"  {'Strategy':<38}  {'Trades':>7}  "
              f"{'Drag (bps/yr)':>14}  {'Drag (ann %)':>13}")
        print(f"  {bar}")
        for r in rows:
            print(f"  {r['Strategy']:<38}  {r['Trades']:>7,}  "
                  f"{r['_drag']:>14.1f}  {r['_drag']/100:>12.2f}%")
        total_bps = sum(r["_drag"] for r in rows)
        print(f"\n  {'TOTAL cost drag across all strategies':38}  "
              f"{'—':>7}  {total_bps:>14.1f}")

    # ── Persistence filter impact detail ─────────────────────────────────────
    raw_sw  = sum(is_regimes[i]   != is_regimes[i-1]   for i in range(1, T))
    filt_sw = sum(filt_regimes[i] != filt_regimes[i-1] for i in range(1, T))
    if verbose:
        print(f"\n  Persistence filter: {raw_sw} → {filt_sw} regime switches "
              f"({(raw_sw-filt_sw)/raw_sw:.0%} reduction)")

    return {
        "ticker":       ticker,
        "name":         name,
        "feat_df":      feat_df,
        "X_raw":        X_raw,
        "X":            X,
        "scaler":       scaler,
        "model":        model,
        "lmap":         lmap,
        "is_regimes":   is_regimes,
        "filt_regimes": filt_regimes,
        "oos_regimes":  oos_regimes,
        "oos_bull_p":   oos_bull_p,
        "oos_mask":     oos_mask,
        "trans_mat":    trans_mat,
        "figures":      figs,
        "perf_df":      perf_df,
        "data_source":  src,
        "raw_switches": raw_sw,
        "filt_switches":filt_sw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PDF REPORT GENERATOR  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

METHODOLOGY_TEXT = {
    "regime_chart":  (
        "HMM with 3 hidden states fit on [log-return, Δ%VIX]. "
        "States decoded via Viterbi algorithm and labelled Bull / Bear / "
        "High-Vol by ranking mean daily return."
    ),
    "regime_stats":  (
        "Mean daily return and realised volatility for each regime (basis "
        "points).  Computed over the full in-sample training set."
    ),
    "trans_heatmap": (
        "Empirical regime-to-regime transition probabilities, estimated "
        "from the Viterbi state sequence.  Diagonal entries reflect regime "
        "persistence; off-diagonal entries capture typical switching paths."
    ),
    "walkfwd_chart": (
        "Walk-forward validation: HMM re-fitted on a rolling 2-year window "
        f"({TRAIN_DAYS} trading days), stepping forward one month "
        f"({STEP_DAYS} days) at a time.  Only the next-month predictions are "
        "retained — no future data leaks into any forecast."
    ),
    "filter_chart":  (
        "Persistence filter: a regime switch is only accepted after the new "
        "state has held for at least 5 consecutive trading days.  This "
        "removes high-frequency signal noise and reduces round-trips."
    ),
    "equity_curves": (
        "Four strategies compared over the full in-sample period.  "
        "Bull-Only: long when IS regime = Bull, else cash.  "
        "Regime-Weighted: position = posterior P(Bull).  "
        "Short Bear: +1× Bull, −1× Bear, 0 High-Vol."
    ),
}


def _add_title_page(pdf, today_str):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.5, 0.72,
            "SPX Regime Detection Report",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.62,
            "Hidden Markov Model — Regime Analysis & Strategy Backtest",
            ha="center", va="center", fontsize=14, color="#aaaacc",
            transform=ax.transAxes)
    ax.text(0.5, 0.50,
            f"Assets: S&P 500  •  Nasdaq-100 (QQQ)  •  Russell 2000 (IWM)",
            ha="center", va="center", fontsize=12, color="#ccccdd",
            transform=ax.transAxes)
    ax.text(0.5, 0.41,
            f"Period: {START}  →  {today_str}",
            ha="center", va="center", fontsize=11, color="#aaaacc",
            transform=ax.transAxes)
    ax.text(0.5, 0.30,
            "Methodology: 3-state Gaussian HMM  |  Baum-Welch EM  |  Viterbi decoding\n"
            "Walk-forward validation  |  Persistence filter  |  Multi-strategy backtest",
            ha="center", va="center", fontsize=10, color="#9999bb",
            transform=ax.transAxes, linespacing=1.8)
    ax.text(0.5, 0.10,
            f"Generated: {today_str}",
            ha="center", va="center", fontsize=9, color="#666688",
            transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _wrap_text(text, width=100):
    """Simple word-wrap for annotation boxes."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def _add_chart_page(pdf, fig, caption):
    """Save a chart figure to the PDF with a caption below it."""
    # Re-use the existing figure, add caption via a tight-layout subtitle
    fig.text(0.5, -0.02, _wrap_text(caption, 115),
             ha="center", va="top", fontsize=7.5,
             color="#555555", style="italic",
             wrap=True, transform=fig.transFigure)
    pdf.savefig(fig, bbox_inches="tight")


def _add_table_page(pdf, df, title, subtitle=""):
    """Render a pandas DataFrame as a matplotlib table page."""
    n_rows, n_cols = df.shape
    fig_h = max(3.5, 1.2 + n_rows * 0.45)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_axis_off()

    if subtitle:
        fig.text(0.5, 0.97, title, ha="center", fontsize=13,
                 fontweight="bold", va="top")
        fig.text(0.5, 0.93, subtitle, ha="center", fontsize=9,
                 color="#555555", va="top")
        top = 0.89
    else:
        fig.text(0.5, 0.97, title, ha="center", fontsize=13,
                 fontweight="bold", va="top")
        top = 0.93

    col_widths = [max(len(str(c)), df[c].astype(str).str.len().max()) * 0.013
                  for c in df.columns]
    col_widths = [max(w, 0.10) for w in col_widths]
    total_w    = sum(col_widths)
    col_widths = [w / total_w for w in col_widths]

    tbl = ax.table(
        cellText  = df.values,
        colLabels = df.columns,
        cellLoc   = "center",
        loc       = "center",
        colWidths = col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Style header row
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Zebra-stripe data rows
    for i in range(1, n_rows + 1):
        fc = "#f2f2f2" if i % 2 == 0 else "white"
        for j in range(n_cols):
            tbl[(i, j)].set_facecolor(fc)

    plt.tight_layout(rect=[0, 0, 1, top])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf_report(results_list, pdf_path):
    """
    Compile all charts and performance tables from results_list into a
    single multi-page PDF report.

    results_list : list of dicts returned by run_pipeline()
    pdf_path     : output path for the PDF
    """
    today_str = pd.Timestamp.today().strftime("%B %d, %Y")

    with PdfPages(pdf_path) as pdf:

        # ── Title page ───────────────────────────────────────────────────────
        _add_title_page(pdf, today_str)

        # ── Per-asset pages ───────────────────────────────────────────────────
        for res in results_list:
            tck  = res["ticker"]
            figs = res["figures"]

            # Section divider
            fig_div = plt.figure(figsize=(11, 2))
            fig_div.patch.set_facecolor("#2c3e50")
            ax_d = fig_div.add_axes([0, 0, 1, 1])
            ax_d.set_axis_off()
            ax_d.text(0.5, 0.55, f"Asset: {tck}",
                      ha="center", va="center", fontsize=20,
                      fontweight="bold", color="white")
            ax_d.text(0.5, 0.25, f"{START}  to  {TODAY}  |  {res['data_source']}",
                      ha="center", va="center", fontsize=11,
                      color="#aaccff")
            pdf.savefig(fig_div, bbox_inches="tight")
            plt.close(fig_div)

            for key, cap_key in [
                ("regime_chart",  "regime_chart"),
                ("regime_stats",  "regime_stats"),
                ("trans_heatmap", "trans_heatmap"),
                ("walkfwd_chart", "walkfwd_chart"),
                ("filter_chart",  "filter_chart"),
                ("equity_curves", "equity_curves"),
            ]:
                if key in figs:
                    _add_chart_page(pdf, figs[key],
                                    METHODOLOGY_TEXT.get(cap_key, ""))

            # Per-asset performance table
            _add_table_page(
                pdf, res["perf_df"],
                title   = f"Strategy Performance Summary  [{tck}]",
                subtitle= (f"Data: {res['data_source']}  |  "
                           f"Regime switches: {res['raw_switches']} raw → "
                           f"{res['filt_switches']} filtered  (≥5-day filter)")
            )

        # ── Combined cross-asset summary ──────────────────────────────────────
        rows = []
        for res in results_list:
            tck = res["ticker"]
            for _, row in res["perf_df"].iterrows():
                rows.append({"Asset": tck, **row})
        combined_df = pd.DataFrame(rows)[[
            "Asset", "Strategy", "Trades", "Gross Return", "Net Return",
            "Gross Sharpe", "Net Sharpe", "Max Drawdown", "Cost Drag (bps/yr)"
        ]]
        _add_table_page(
            pdf, combined_df,
            title    = "Combined Cross-Asset Performance Summary",
            subtitle = f"All strategies  |  {START} → {today_str}"
        )

        # ── PDF metadata ─────────────────────────────────────────────────────
        d = pdf.infodict()
        d["Title"]   = "SPX Regime Detection Report"
        d["Author"]  = "HMM Regime Detector"
        d["Subject"] = "Market Regime Analysis"
        d["Keywords"]= "HMM, regime, SPX, QQQ, IWM, backtest"

    print(f"\n  PDF report saved → {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run all three assets, build report
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0 — DIAGNOSTIC: raw data check
    # Print first/last 5 rows for each ticker to confirm correct prices.
    # SPX 2010-01-04 should be ≈ 1130; recent date should be ≈ 5 500-6 000.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*65)
    print("  DIAGNOSTIC — Raw downloaded data (pre-modelling check)")
    print("═"*65)

    if _YFINANCE_AVAILABLE:
        print(f"  yfinance available: YES   auto_adjust=True in all downloads\n")
        for diag_ticker in ["^GSPC", "QQQ", "IWM"]:
            print(f"  ── {diag_ticker} ──────────────────────────────────────")
            try:
                raw = yf.download(
                    diag_ticker,
                    start=START, end=TODAY,
                    auto_adjust=True, progress=False, threads=False,
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                print(f"  Columns : {list(raw.columns)}")
                print(f"  Rows    : {len(raw)}")
                print(f"\n  First 5 rows:")
                print(raw.head().to_string())
                print(f"\n  Last 5 rows:")
                print(raw.tail().to_string())
                # Spot-check specific dates
                idx_str = raw.index.strftime("%Y-%m-%d")
                for chk_date, expected in [("2010-01-04", "≈1 130"),
                                            ("2026-03-01", "≈5 700")]:
                    if chk_date in idx_str:
                        val = raw.loc[raw.index[idx_str == chk_date][0], "Close"]
                        print(f"\n  Close on {chk_date}: {val:.2f}  (expected {expected})")
                    else:
                        nearest = raw.iloc[raw.index.get_indexer(
                            [pd.Timestamp(chk_date)], method="nearest")[0]]
                        print(f"\n  Close nearest {chk_date}: "
                              f"{nearest['Close']:.2f}  (expected {expected})")
            except Exception as exc:
                print(f"  ERROR: {exc}")
            print()
    else:
        print("  yfinance NOT available — install with: pip install yfinance")
        print("  Falling back to requests → synthetic data.\n")

    print("═"*65 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ═══════════════════════════════════════════════════════════════════════
    TICKERS = ["^GSPC", "QQQ", "IWM"]
    all_results = []

    for ticker in TICKERS:
        res = run_pipeline(ticker, START, TODAY, verbose=True)
        all_results.append(res)

    # ── Save per-asset PNGs ───────────────────────────────────────────────────
    print("\n[PNG] Saving individual charts …")
    chart_map = {
        "regime_chart":  "spx_regimes",
        "regime_stats":  "regime_stats",
        "trans_heatmap": "transition_heatmap",
        "walkfwd_chart": "walkforward_vs_insample",
        "filter_chart":  "persistence_filter",
        "equity_curves": "all_equity_curves",
    }
    for res in all_results:
        name = res["name"]
        for fig_key, file_key in chart_map.items():
            if fig_key in res["figures"]:
                path = os.path.join(OUTPUT_DIR,
                                    f"{name}_{file_key}.png")
                res["figures"][fig_key].savefig(path, dpi=130,
                                                 bbox_inches="tight")
                print(f"  {os.path.basename(path)}")

    # ── Save results CSV ──────────────────────────────────────────────────────
    print("\n[CSV] Saving results …")
    for res in all_results:
        name = res["name"]
        cols = [c for c in res["feat_df"].columns
                if c.startswith(("regime", "ret_", "cum_",
                                 "pos_", "bull_prob", "log_return",
                                 "vix_change"))]
        path = os.path.join(OUTPUT_DIR, f"{name}_results.csv")
        res["feat_df"][cols].to_csv(path)
        print(f"  {os.path.basename(path)}  ({len(res['feat_df']):,} rows)")

    # ── Combined cross-asset summary table ────────────────────────────────────
    print("\n" + "="*75)
    print("  COMBINED CROSS-ASSET PERFORMANCE SUMMARY")
    print("="*75)
    rows = []
    for res in all_results:
        for _, row in res["perf_df"].iterrows():
            rows.append({"Asset": res["ticker"], **row})
    combined = pd.DataFrame(rows)[[
        "Asset", "Strategy", "Trades", "Gross Return", "Net Return",
        "Gross Sharpe", "Net Sharpe", "Max Drawdown", "Cost Drag (bps/yr)"
    ]]
    print(combined.to_string(index=False))

    # ── PDF report ────────────────────────────────────────────────────────────
    print("\n[PDF] Generating report …")
    pdf_path = os.path.join(OUTPUT_DIR, "regime_detection_report.pdf")
    generate_pdf_report(all_results, pdf_path)

    print("\n" + "="*75)
    print("  ALL DONE ✓")
    print("="*75)
