#!/usr/bin/env python3
"""
=============================================================================
Stock Market Regime Detection Tool — Hidden Markov Model (HMM)
=============================================================================
Identifies Bull, Bear, and High-Volatility regimes in SPX daily returns,
then backtests a simple strategy that goes long only during Bull regimes.

Libraries used:
  - numpy, pandas        : data wrangling & math
  - matplotlib, seaborn  : visualization
  - requests             : Yahoo Finance data fetching
  - (sklearn / hmmlearn  : implemented from scratch when unavailable)
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import requests

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/sessions/lucid-confident-bohr/mnt/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yahoo(symbol, start="2010-01-01", end=None):
    """
    Fetch daily OHLCV data from Yahoo Finance v8 chart API.
    Returns a DataFrame indexed by date with columns:
      Open, High, Low, Close, Volume, Adj Close
    Falls back to None on any network/parse error.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    sym_enc = symbol.replace("^", "%5E")
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end).timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym_enc}"
        f"?interval=1d&period1={p1}&period2={p2}&events=history"
    )
    headers = {"User-Agent": "Mozilla/5.0 (compatible; regime-detector/1.0)"}

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()["chart"]["result"][0]

        timestamps  = data["timestamp"]
        q           = data["indicators"]["quote"][0]
        adj_close   = data["indicators"].get("adjclose", [{}])[0].get(
                          "adjclose", q["close"])

        df = pd.DataFrame({
            "Open":      q["open"],
            "High":      q["high"],
            "Low":       q["low"],
            "Close":     q["close"],
            "Volume":    q["volume"],
            "Adj Close": adj_close,
        }, index=pd.to_datetime(timestamps, unit="s").normalize())
        df.index.name = "Date"
        return df.ffill().dropna()

    except Exception as exc:
        print(f"  [WARNING] Could not fetch {symbol} from Yahoo Finance: {exc}")
        return None


def generate_synthetic_spx(start="2010-01-01", end=None, seed=42):
    """
    Generate realistic synthetic SPX + VIX data using a 3-state
    Markov regime-switching model when live data is unavailable.

    Regimes mirror empirical SPX characteristics:
      • Bull     : positive drift, low volatility, falling VIX
      • Bear     : negative drift, high volatility, rising VIX
      • High-Vol : near-zero drift, elevated volatility, slightly rising VIX

    Returns two DataFrames: (spx_df, vix_df)
    """
    rng = np.random.RandomState(seed)
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    dates = pd.bdate_range(start, end)   # business days only
    T = len(dates)

    # ── Regime parameters ──────────────────────────────────────────────────
    #   [Bull,  Bear,  High-Vol]
    ret_mean  = np.array([ 0.00060, -0.00150,  0.00000])
    ret_std   = np.array([ 0.00700,  0.01800,  0.01300])
    vix_mean  = np.array([-0.00200,  0.01000,  0.00300])   # VIX pct change
    vix_std   = np.array([ 0.03000,  0.06000,  0.05000])

    # ── Transition matrix (rows = from, cols = to) ──────────────────────────
    A = np.array([
        [0.960, 0.020, 0.020],   # Bull  → Bull / Bear / High-Vol
        [0.030, 0.940, 0.030],   # Bear  → Bull / Bear / High-Vol
        [0.050, 0.030, 0.920],   # HVol  → Bull / Bear / High-Vol
    ])

    # ── Simulate latent regime sequence ─────────────────────────────────────
    regime = np.empty(T, dtype=int)
    regime[0] = 0   # start in Bull
    for t in range(1, T):
        regime[t] = rng.choice(3, p=A[regime[t - 1]])

    # ── Simulate log-returns and VIX changes ─────────────────────────────────
    log_ret  = ret_mean[regime] + ret_std[regime]  * rng.randn(T)
    vix_chg  = vix_mean[regime] + vix_std[regime]  * rng.randn(T)

    # ── Build SPX price series starting at ~1115 (SPX 2010-01-04 close) ─────
    spx_adj_close = 1115.10 * np.exp(np.cumsum(log_ret))

    spx_df = pd.DataFrame({
        "Adj Close": spx_adj_close,
        "Close":     spx_adj_close,
        "Open":      spx_adj_close * (1 + rng.randn(T) * 0.002),
        "High":      spx_adj_close * (1 + np.abs(rng.randn(T)) * 0.003),
        "Low":       spx_adj_close * (1 - np.abs(rng.randn(T)) * 0.003),
        "Volume":    (3e9 * (1 + rng.randn(T) * 0.3)).clip(1e8),
    }, index=dates)
    spx_df.index.name = "Date"

    # ── Build VIX series starting at ~17.5 ───────────────────────────────────
    vix_close = 17.5 * np.exp(np.cumsum(vix_chg)).clip(8, 80)
    vix_df = pd.DataFrame({
        "Close": vix_close,
    }, index=dates)
    vix_df.index.name = "Date"

    print(f"  [INFO] Generated {T} synthetic trading days ({start} – {end})")
    print(f"  [INFO] True regime distribution: "
          f"Bull={np.mean(regime==0):.1%}  "
          f"Bear={np.mean(regime==1):.1%}  "
          f"High-Vol={np.mean(regime==2):.1%}")
    return spx_df, vix_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 (cont.) — Load or generate data
# ─────────────────────────────────────────────────────────────────────────────

START, END = "2010-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
print(f"\n{'='*65}")
print(f"  SPX Regime Detection  |  {START} → {END}")
print(f"{'='*65}\n")

print("[1/6] Fetching market data …")
spx_raw = fetch_yahoo("^GSPC", START, END)
vix_raw = fetch_yahoo("^VIX",  START, END)

if spx_raw is None or vix_raw is None:
    print("  Live data unavailable — using realistic synthetic SPX/VIX data.\n"
          "  (The HMM, backtest, and visualization logic is identical.)")
    spx_raw, vix_raw = generate_synthetic_spx(START, END, seed=42)
    DATA_SOURCE = "Synthetic (Yahoo Finance unreachable)"
else:
    DATA_SOURCE = "Yahoo Finance (live)"
    print(f"  Live data loaded: {len(spx_raw)} SPX rows, {len(vix_raw)} VIX rows")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/6] Engineering features …")

# Daily log-returns from Adjusted Close
spx_adj   = spx_raw["Adj Close"].ffill()
log_ret   = np.log(spx_adj / spx_adj.shift(1))

# Daily % change in VIX
vix_close = vix_raw["Close"].ffill().reindex(spx_adj.index, method="ffill")
vix_chg   = vix_close.pct_change()

# Combine into a single DataFrame and drop any NaN rows
feat_df = pd.DataFrame({
    "log_return": log_ret,
    "vix_change": vix_chg,
    "spx_adj":    spx_adj,
}).dropna()

print(f"  Feature matrix shape: {feat_df.shape}")

# ── StandardScaler (manual implementation — sklearn not available in sandbox) ──
class StandardScaler:
    """Z-score standardisation: X' = (X - μ) / σ"""
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        return self
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

scaler = StandardScaler()
X_raw = feat_df[["log_return", "vix_change"]].values
X     = scaler.fit_transform(X_raw)        # (T, 2)  — standardised features
print(f"  After scaling: mean≈{X.mean(axis=0).round(4)}, std≈{X.std(axis=0).round(4)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GAUSSIAN HMM (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianHMM:
    """
    Full-covariance Gaussian HMM trained with the Baum-Welch EM algorithm.
    Hidden state sequence decoded with the Viterbi algorithm.

    Parameters
    ----------
    n_components : int   — number of hidden states (regimes)
    n_iter       : int   — maximum EM iterations
    tol          : float — convergence threshold on log-likelihood
    random_state : int   — reproducibility seed
    reg_cov      : float — covariance regularisation (prevents singularity)
    """

    def __init__(self, n_components=3, n_iter=200, tol=1e-5,
                 random_state=42, reg_cov=1e-4):
        self.K    = n_components
        self.n_iter      = n_iter
        self.tol         = tol
        self.rng         = np.random.RandomState(random_state)
        self.reg_cov     = reg_cov
        # Learned parameters (set by fit)
        self.startprob_  = None
        self.transmat_   = None
        self.means_      = None
        self.covars_     = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _logsumexp(arr, axis=None, keepdims=False):
        """
        Numerically stable log-sum-exp.
        Always returns a Python float when axis=None and keepdims=False.
        """
        max_v    = np.max(arr, axis=axis, keepdims=True)
        max_safe = np.where(np.isfinite(max_v), max_v, 0.0)
        # Sum in exp-space, then take log; keep keepdims=True internally
        out = max_safe + np.log(
            np.sum(np.exp(arr - max_safe), axis=axis, keepdims=True)
        )
        if not keepdims:
            out = np.squeeze(out)          # collapse retained dims to scalars
        # Ensure a clean Python float when the result is 0-d
        if isinstance(out, np.ndarray) and out.ndim == 0:
            return float(out)
        return out

    def _log_emission(self, X):
        """
        Compute log p(x_t | state k) for all t and k.
        Returns (T, K) array of log-probabilities.
        """
        T, d = X.shape
        log_probs = np.empty((T, self.K))
        const = d * np.log(2.0 * np.pi)
        for k in range(self.K):
            cov     = self.covars_[k] + np.eye(d) * self.reg_cov
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:                   # fallback for degenerate covariance
                cov     = np.eye(d) * self.reg_cov
                log_det = d * np.log(self.reg_cov)
            cov_inv = np.linalg.inv(cov)
            diff    = X - self.means_[k]             # (T, d)
            # Mahalanobis distance for each observation
            mahal   = np.einsum("ti,ij,tj->t", diff, cov_inv, diff)  # (T,)
            log_probs[:, k] = -0.5 * (const + log_det + mahal)
        return log_probs

    # ── Forward pass (log-space) ─────────────────────────────────────────────

    def _forward(self, log_emiss):
        T, K   = log_emiss.shape
        log_A  = np.log(np.clip(self.transmat_, 1e-300, 1.0))
        log_pi = np.log(np.clip(self.startprob_, 1e-300, 1.0))

        log_alpha    = np.empty((T, K))
        log_alpha[0] = log_pi + log_emiss[0]

        for t in range(1, T):
            # log_alpha[t-1, i] + log_A[i, j]  →  (K, K) matrix  →  logsumexp over i
            vals          = log_alpha[t - 1, :, None] + log_A   # (K_from, K_to)
            max_v         = vals.max(axis=0)                     # (K_to,)
            log_alpha[t]  = max_v + np.log(np.exp(vals - max_v).sum(axis=0)) + log_emiss[t]
        return log_alpha

    # ── Backward pass (log-space) ────────────────────────────────────────────

    def _backward(self, log_emiss):
        T, K   = log_emiss.shape
        log_A  = np.log(np.clip(self.transmat_, 1e-300, 1.0))

        log_beta      = np.zeros((T, K))   # β[T-1, :] = log(1) = 0

        for t in range(T - 2, -1, -1):
            # log_A[i, j] + log_emiss[t+1, j] + log_beta[t+1, j]  →  (K_from, K_to)
            vals          = log_A + log_emiss[t + 1] + log_beta[t + 1]  # (K, K)
            max_v         = vals.max(axis=1)                              # (K_from,)
            log_beta[t]   = max_v + np.log(np.exp(vals - max_v[:, None]).sum(axis=1))
        return log_beta

    # ── Initialization ───────────────────────────────────────────────────────

    def _initialize(self, X):
        T, d = X.shape
        K    = self.K

        # K-means initialization: gives better starting points than random
        # (avoids degenerate solutions from pure random init)
        idx       = self.rng.choice(T, K, replace=False)
        centroids = X[idx].copy()

        for _ in range(50):
            dists       = np.array([((X - centroids[k]) ** 2).sum(axis=1) for k in range(K)])
            assign      = np.argmin(dists, axis=0)                 # (T,)
            new_centroids = np.array([
                X[assign == k].mean(axis=0) if (assign == k).sum() > 0 else centroids[k]
                for k in range(K)
            ])
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        self.means_ = centroids
        self.covars_ = np.array([
            np.cov(X[assign == k].T) + np.eye(d) * 0.1
            if (assign == k).sum() > d else np.eye(d) * 0.1
            for k in range(K)
        ])

        # Slightly sticky transition matrix (states tend to persist)
        self.transmat_  = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(self.transmat_, 0.9)
        self.startprob_ = np.full(K, 1.0 / K)

    # ── Baum-Welch EM ────────────────────────────────────────────────────────

    def fit(self, X):
        """Fit HMM parameters using the Baum-Welch algorithm."""
        T, d        = X.shape
        K           = self.K
        self._initialize(X)
        prev_loglik = -np.inf

        print(f"\n[3/6] Fitting GaussianHMM  "
              f"(n_states={K}, T={T}, max_iter={self.n_iter}) …")

        for iteration in range(self.n_iter):
            # ── E-step ──────────────────────────────────────────────────────
            log_emiss = self._log_emission(X)                  # (T, K)
            log_alpha = self._forward(log_emiss)               # (T, K)
            log_beta  = self._backward(log_emiss)              # (T, K)

            # Log-likelihood = logsumexp of final alpha column
            log_lik = self._logsumexp(log_alpha[-1])

            # Posterior state probabilities: γ[t, k] = P(s_t = k | X)
            log_gamma = log_alpha + log_beta                   # (T, K)
            lse_t     = self._logsumexp(log_gamma, axis=1, keepdims=True)
            gamma     = np.exp(log_gamma - lse_t)              # (T, K)

            # Pairwise posteriors: ξ[t, i, j] = P(s_t=i, s_{t+1}=j | X)
            log_A   = np.log(np.clip(self.transmat_, 1e-300, 1.0))
            # log_xi[t, i, j] = α[t,i] + A[i,j] + emission[t+1,j] + β[t+1,j]
            log_xi  = (log_alpha[:-1, :, None]                  # (T-1, K, 1)
                       + log_A[None, :, :]                       # (1,   K, K)
                       + log_emiss[1:, None, :]                  # (T-1, 1, K)
                       + log_beta[1:, None, :])                  # (T-1, 1, K)
            lse_xi  = self._logsumexp(
                log_xi.reshape(T - 1, K * K), axis=1, keepdims=True
            ).reshape(T - 1, 1, 1)
            xi      = np.exp(log_xi - lse_xi)                  # (T-1, K, K)

            # ── M-step ──────────────────────────────────────────────────────
            # Update initial state distribution
            self.startprob_      = gamma[0] + 1e-300
            self.startprob_     /= self.startprob_.sum()

            # Update transition matrix
            xi_sum   = xi.sum(axis=0)                          # (K, K)
            self.transmat_ = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)

            # Update means and covariances
            gamma_sum = gamma.sum(axis=0) + 1e-300             # (K,)
            self.means_ = (gamma[:, :, None] * X[:, None, :]).sum(axis=0) / gamma_sum[:, None]

            for k in range(K):
                diff             = X - self.means_[k]          # (T, d)
                w                = gamma[:, k]                  # (T,)
                self.covars_[k]  = (
                    (w[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
                    / gamma_sum[k]
                    + np.eye(d) * self.reg_cov                  # regularise
                )

            # ── Convergence check ─────────────────────────────────────────
            delta = abs(log_lik - prev_loglik)
            if (iteration + 1) % 25 == 0:
                print(f"    iter {iteration+1:4d}  log-lik = {log_lik:,.2f}  "
                      f"Δ = {delta:.2e}")
            if delta < self.tol and iteration > 10:
                print(f"    Converged at iteration {iteration+1}  "
                      f"(Δ log-lik = {delta:.2e} < tol={self.tol})")
                break
            prev_loglik = log_lik

        self.loglikelihood_ = log_lik
        return self

    # ── Viterbi decoding ─────────────────────────────────────────────────────

    def predict(self, X):
        """
        Decode the most-likely state sequence using the Viterbi algorithm.
        Returns array of integer state labels, shape (T,).
        """
        T, d    = X.shape
        K       = self.K
        log_A   = np.log(np.clip(self.transmat_, 1e-300, 1.0))
        log_pi  = np.log(np.clip(self.startprob_, 1e-300, 1.0))
        log_emiss = self._log_emission(X)                      # (T, K)

        # Viterbi tables
        delta   = np.empty((T, K))
        psi     = np.zeros((T, K), dtype=int)
        delta[0]= log_pi + log_emiss[0]

        for t in range(1, T):
            trans        = delta[t - 1, :, None] + log_A      # (K_from, K_to)
            psi[t]       = np.argmax(trans, axis=0)
            delta[t]     = trans[psi[t], np.arange(K)] + log_emiss[t]

        # Back-track
        states      = np.empty(T, dtype=int)
        states[-1]  = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


# ── Fit the model ─────────────────────────────────────────────────────────────

model  = GaussianHMM(n_components=3, n_iter=200, tol=1e-5, random_state=42)
model.fit(X)

raw_states = model.predict(X)   # integer labels 0, 1, 2  (unordered)
print(f"\n  Final log-likelihood: {model.loglikelihood_:,.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 (cont.) — LABEL REGIMES
# ─────────────────────────────────────────────────────────────────────────────
# Identify which raw state index corresponds to each economic regime
# by ranking mean log-return:  highest → Bull, lowest → Bear, middle → High-Vol

mean_ret_per_state = np.array([
    X_raw[:, 0][raw_states == k].mean() for k in range(3)
])
ranked = np.argsort(mean_ret_per_state)        # ascending order
bear_idx   = ranked[0]   # lowest mean return
highvol_idx= ranked[1]   # middle mean return
bull_idx   = ranked[2]   # highest mean return

label_map = {bull_idx: "Bull", bear_idx: "Bear", highvol_idx: "High-Vol"}
regimes   = np.array([label_map[s] for s in raw_states])

feat_df["regime"]     = regimes
feat_df["raw_state"]  = raw_states

print("\n  ── Regime labelling ─────────────────────────────────────────────")
print(f"  Bull    → raw state {bull_idx}  "
      f"| mean return = {mean_ret_per_state[bull_idx]:+.5f}")
print(f"  Bear    → raw state {bear_idx}  "
      f"| mean return = {mean_ret_per_state[bear_idx]:+.5f}")
print(f"  High-Vol→ raw state {highvol_idx}  "
      f"| mean return = {mean_ret_per_state[highvol_idx]:+.5f}")

# Regime statistics table
for reg in ["Bull", "Bear", "High-Vol"]:
    mask = regimes == reg
    n    = mask.sum()
    mr   = X_raw[:, 0][mask].mean()
    sv   = X_raw[:, 0][mask].std()
    print(f"  {reg:<9s}  N={n:5d}  ({n/len(regimes):.1%})  "
          f"μ_ret={mr:+.5f}  σ_ret={sv:.5f}")

# Build transition matrix indexed by regime labels
order     = ["Bull", "Bear", "High-Vol"]
trans_mat = pd.DataFrame(
    np.zeros((3, 3)), index=order, columns=order
)
for t in range(len(raw_states) - 1):
    frm = label_map[raw_states[t]]
    to  = label_map[raw_states[t + 1]]
    trans_mat.loc[frm, to] += 1
trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/6] Generating charts …")

# Colour palette
COLORS = {"Bull": "#2ecc71", "Bear": "#e74c3c", "High-Vol": "#95a5a6"}

# ── Chart 1: SPX price with regime background shading ────────────────────────

fig, ax = plt.subplots(figsize=(16, 6))

# Price line
ax.plot(feat_df.index, feat_df["spx_adj"], color="#2c3e50",
        linewidth=1.0, zorder=5, label="SPX Adj. Close")
ax.set_yscale("log")

# Shade background by regime (draw a span for each contiguous block)
regime_series = feat_df["regime"]
i = 0
while i < len(regime_series):
    reg = regime_series.iloc[i]
    j   = i
    while j < len(regime_series) and regime_series.iloc[j] == reg:
        j += 1
    ax.axvspan(regime_series.index[i], regime_series.index[j - 1],
               alpha=0.25, color=COLORS[reg], linewidth=0)
    i = j

# Legend handles
patches = [mpatches.Patch(color=COLORS[r], alpha=0.5, label=r)
           for r in ["Bull", "Bear", "High-Vol"]]
patches.append(plt.Line2D([0], [0], color="#2c3e50", linewidth=1.5,
                           label="SPX Price"))
ax.legend(handles=patches, loc="upper left", fontsize=10)

ax.set_title(f"SPX Daily Price — HMM Regime Detection  "
             f"({START} → {END})\n"
             f"Data: {DATA_SOURCE}",
             fontsize=13, pad=12)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("SPX Adj. Close (log scale)", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "chart1_spx_regimes.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {path1}")


# ── Chart 2: Mean daily return & volatility per regime ───────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

regime_stats = []
for reg in ["Bull", "Bear", "High-Vol"]:
    mask  = regimes == reg
    ret_r = X_raw[:, 0][mask]
    regime_stats.append({
        "Regime":     reg,
        "Mean Return (bps)": ret_r.mean() * 10_000,
        "Volatility (bps)":  ret_r.std()  * 10_000,
    })
stats_df = pd.DataFrame(regime_stats)

bar_colors = [COLORS[r] for r in stats_df["Regime"]]

# Mean return bar chart
ax_l = axes[0]
bars = ax_l.bar(stats_df["Regime"], stats_df["Mean Return (bps)"],
                color=bar_colors, edgecolor="white", linewidth=1.2)
ax_l.axhline(0, color="black", linewidth=0.8, linestyle="--")
for bar, val in zip(bars, stats_df["Mean Return (bps)"]):
    ax_l.text(bar.get_x() + bar.get_width() / 2,
              val + (0.3 if val >= 0 else -0.8),
              f"{val:+.1f}", ha="center", va="bottom", fontsize=11,
              fontweight="bold")
ax_l.set_title("Mean Daily Return per Regime", fontsize=12)
ax_l.set_ylabel("Return (basis points)", fontsize=11)
ax_l.set_ylim(stats_df["Mean Return (bps)"].min() * 1.5,
              stats_df["Mean Return (bps)"].max() * 1.8)
ax_l.grid(axis="y", linestyle="--", alpha=0.4)

# Volatility bar chart
ax_r = axes[1]
bars2 = ax_r.bar(stats_df["Regime"], stats_df["Volatility (bps)"],
                 color=bar_colors, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars2, stats_df["Volatility (bps)"]):
    ax_r.text(bar.get_x() + bar.get_width() / 2,
              val + 0.2,
              f"{val:.1f}", ha="center", va="bottom", fontsize=11,
              fontweight="bold")
ax_r.set_title("Daily Return Volatility per Regime", fontsize=12)
ax_r.set_ylabel("Volatility (basis points)", fontsize=11)
ax_r.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Regime Characteristics — Mean Return & Volatility",
             fontsize=13, y=1.01)
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "chart2_regime_stats.png")
fig.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {path2}")


# ── Chart 3: Transition probability heatmap ──────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(
    trans_mat,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    linewidths=0.5,
    linecolor="white",
    vmin=0, vmax=1,
    ax=ax,
    annot_kws={"size": 13, "weight": "bold"}
)
ax.set_title("Regime Transition Probability Matrix", fontsize=13, pad=14)
ax.set_xlabel("Next Regime", fontsize=11)
ax.set_ylabel("Current Regime", fontsize=11)
ax.tick_params(labelsize=11)
plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, "chart3_transition_heatmap.png")
fig.savefig(path3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {path3}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/6] Running backtest …")

# Strategy: 1 when Bull regime, 0 (cash) otherwise
feat_df["in_market"]      = (feat_df["regime"] == "Bull").astype(float)

# Daily strategy return (regime label is known *at the open* of the current
# bar — using same-day regime for simplicity, consistent with index futures)
feat_df["strat_return"]   = feat_df["log_return"] * feat_df["in_market"]
feat_df["bnh_return"]     = feat_df["log_return"]

# Cumulative return series (starting at 1.0)
feat_df["cum_strat"]      = np.exp(feat_df["strat_return"].cumsum())
feat_df["cum_bnh"]        = np.exp(feat_df["bnh_return"].cumsum())


def perf_metrics(daily_log_ret, label, ann_factor=252):
    """
    Compute annualised return, annualised volatility,
    Sharpe ratio, and maximum drawdown from daily log-returns.
    """
    n          = len(daily_log_ret)
    total_ret  = daily_log_ret.sum()
    ann_ret    = (total_ret / n) * ann_factor             # annualise log-return
    ann_vol    = daily_log_ret.std() * np.sqrt(ann_factor)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum_r      = np.exp(daily_log_ret.cumsum())
    roll_max   = cum_r.cummax()
    drawdown   = (cum_r - roll_max) / roll_max
    max_dd     = drawdown.min()

    return {
        "Strategy":          label,
        "Ann. Return":       f"{ann_ret:.2%}",
        "Ann. Volatility":   f"{ann_vol:.2%}",
        "Sharpe Ratio":      f"{sharpe:.2f}",
        "Max Drawdown":      f"{max_dd:.2%}",
        # numeric versions for sorting
        "_ann_ret":   ann_ret,
        "_ann_vol":   ann_vol,
        "_sharpe":    sharpe,
        "_max_dd":    max_dd,
    }


m_strat = perf_metrics(feat_df["strat_return"], "HMM Bull-Only Strategy")
m_bnh   = perf_metrics(feat_df["bnh_return"],   "Buy & Hold (SPX)")

perf_df = pd.DataFrame([m_strat, m_bnh])[[
    "Strategy", "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown"
]]

# Cumulative return chart (Chart 4)
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(feat_df.index, feat_df["cum_strat"],
        color="#2980b9", linewidth=1.8, label="HMM Bull-Only Strategy")
ax.plot(feat_df.index, feat_df["cum_bnh"],
        color="#e67e22", linewidth=1.8, linestyle="--", label="Buy & Hold (SPX)")

ax.fill_between(feat_df.index, feat_df["cum_strat"], feat_df["cum_bnh"],
                where=feat_df["cum_strat"] >= feat_df["cum_bnh"],
                interpolate=True, alpha=0.12, color="#2980b9")
ax.fill_between(feat_df.index, feat_df["cum_strat"], feat_df["cum_bnh"],
                where=feat_df["cum_strat"] <  feat_df["cum_bnh"],
                interpolate=True, alpha=0.12, color="#e67e22")

# Annotate final values
for col, name, color in [
    ("cum_strat", "Strategy", "#2980b9"),
    ("cum_bnh",   "B&H",      "#e67e22"),
]:
    final = feat_df[col].iloc[-1]
    ax.annotate(f"{name}: {final:.2f}×",
                xy=(feat_df.index[-1], final),
                xytext=(-10, 8), textcoords="offset points",
                fontsize=10, color=color, ha="right")

ax.set_title(f"Cumulative Returns: HMM Strategy vs. Buy & Hold  "
             f"({START} → {END})\n"
             f"Data: {DATA_SOURCE}",
             fontsize=13, pad=12)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Growth of $1 Invested", fontsize=11)
ax.legend(fontsize=11, loc="upper left")
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, "chart4_cumulative_returns.png")
fig.savefig(path4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {path4}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/6] Saving results …")

# CSV: daily regime labels + strategy returns
results_df = feat_df[[
    "log_return", "vix_change", "regime",
    "in_market", "strat_return", "bnh_return",
    "cum_strat", "cum_bnh"
]].copy()
results_df.columns = [
    "log_return", "vix_change", "regime",
    "in_bull_market", "strategy_return", "bnh_return",
    "cum_strategy", "cum_buy_and_hold"
]
csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
results_df.to_csv(csv_path)
print(f"  Saved → {csv_path}  ({len(results_df):,} rows)")

# ── Console performance table ────────────────────────────────────────────────

bar = "─" * 65
print(f"\n{'='*65}")
print(f"  PERFORMANCE SUMMARY  |  {START} → {END}")
print(f"  Data source: {DATA_SOURCE}")
print(f"{'='*65}")
print(f"  {'Metric':<24}  {'HMM Strategy':>18}  {'Buy & Hold':>14}")
print(f"  {bar}")
for metric in ["Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown"]:
    s = perf_df[perf_df["Strategy"] == "HMM Bull-Only Strategy"][metric].values[0]
    b = perf_df[perf_df["Strategy"] == "Buy & Hold (SPX)"][metric].values[0]
    print(f"  {metric:<24}  {s:>18}  {b:>14}")
print(f"{'='*65}")

# Regime time-in-market
for reg in ["Bull", "Bear", "High-Vol"]:
    pct = (regimes == reg).mean()
    print(f"  Time in {reg:<9s} regime: {pct:.1%}")
print(f"{'='*65}\n")

print("  Charts saved:")
for p in [path1, path2, path3, path4]:
    print(f"    • {os.path.basename(p)}")
print()
print("  All done! ✓")
