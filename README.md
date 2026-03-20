# HMM Regime Detection

A 3-state Gaussian Hidden Markov Model for detecting Bull, Bear, and 
High-Volatility regimes in US equity markets, with walk-forward 
out-of-sample validation and transaction cost modeling.

## Results Summary

| Asset | Strategy | Net Return | Net Sharpe | Max Drawdown | Trades |
|-------|----------|-----------|------------|--------------|--------|
| SPX | Buy & Hold | 3.03% | 0.18 | -54.65% | 0 |
| SPX | Bull-Only (IS) | 5.16% | 0.68 | -17.33% | 85 |
| SPX | Persistence Filter | 5.37% | 0.68 | -17.33% | 65 |
| SPX | Bull-Only (OOS) | 5.68% | 0.64 | -26.36% | 333 |
| QQQ | Bull-Only (OOS) | 7.08% | 0.74 | -22.13% | 325 |
| IWM | Bull-Only (OOS) | 12.29% | 1.29 | -15.01% | 357 |

*Net of 5 bps/trade transaction costs. Period: 2010–2026.*

## Methodology

**Model:** 3-state Gaussian HMM fit on `[log_return, Δ%VIX]` daily 
features using Baum-Welch EM. States decoded via Viterbi algorithm and 
labelled Bull / Bear / High-Vol by ranking mean daily return.

**Validation:** Walk-forward OOS — HMM re-fitted on a rolling 2-year 
window, stepping forward 1 month at a time. Only next-month predictions 
retained (no future data leakage).

**Persistence filter:** Regime switches accepted only after the new state 
holds for ≥5 consecutive trading days, reducing noise and round-trips.

**Transaction costs:** 5 bps per trade (one-way). Regime-Weighted strategy 
uses a ≥5% position change threshold to avoid excessive turnover.

**Strategies tested:**
- Bull-Only: long when regime = Bull, else cash
- Persistence Filter: Bull-Only with 5-day confirmation lag
- Regime-Weighted: position size = posterior P(Bull), ≥5% change threshold
- Short Bear: +1× Bull, −1× Bear, 0× High-Vol

## Key Findings

- Bull regime: +4.9 bps/day mean return, 71.8 bps daily vol
- Bear regime: −6.3 bps/day mean return, 165.5 bps daily vol  
- Regime persistence: 97.7% probability of staying in current regime day-to-day
- All regime strategies beat buy-and-hold on Sharpe and max drawdown
- Persistence Filter achieves similar Sharpe to Bull-Only with 24% fewer trades

## Limitations

- Data sourced via yfinance (calibrated synthetic prices — verify before 
  live use)
- All IS strategies use in-sample regime labels (data leakage — use OOS 
  results for honest evaluation)
- Short Bear IS performance is inflated by in-sample regime knowledge
- No slippage, market impact, or borrowing costs modeled
- HMM is sensitive to initialisation — results may vary across random seeds

## Setup
```bash
git clone https://github.com/sakshammgarg/HMM_Regime_Detection
cd HMM_Regime_Detection
pip install -r requirements.txt
python src/hmm_regime_extended.py
```

## Requirements

See `requirements.txt`. Main dependencies: `yfinance`, `hmmlearn`, 
`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

## Output

Running the script produces all figures in `Outputs/figures/`, CSVs in 
`Outputs/data/`, and a full PDF report at `Outputs/regime_detection_report.pdf`.
```

**`requirements.txt`:**
```
yfinance>=0.2.36
hmmlearn>=0.3.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
reportlab>=4.0.0
```

**`.gitignore`:**
```
__pycache__/
*.pyc
.DS_Store
.env
*.egg-info/
dist/
.ipynb_checkpoints/
