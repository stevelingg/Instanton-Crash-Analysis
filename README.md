# Instanton Crash Analysis

Rare-event drawdown forecasting in US equities using **instanton (minimum-action) paths** and **importance sampling** for daily crash probabilities.

This repository implements an end-to-end pipeline that:

- builds a daily state from SPY prices,
- fits a 2D Markov stochastic-volatility (SV) diffusion on a matched daily discretisation,
- computes fixed-time instantons for finite-horizon drawdown-threshold hitting events,
- uses the instanton to construct an importance sampler with exact Gaussian-shift likelihood ratios,
- produces daily crash probabilities together with Monte Carlo uncertainty diagnostics,
- evaluates forecasts with scoring rules, calibration diagnostics, and optional naive-Monte-Carlo checks.

---

## Why this repository exists

Extreme market crashes are difficult to estimate with standard simulation because the events of interest are rare and path-dependent. In volatility-based models, it is also easy to capture clustering in variance without correctly capturing the structure of the far tail. This project explores a different viewpoint: instead of treating a crash as “just a very large return”, it treats a crash as a **rare trajectory** of a stochastic system.

The practical idea is:

1. define a finite-horizon drawdown event on a daily grid,
2. compute the **minimum-action path** that reaches that event,
3. use that path to bias simulation toward crash-like trajectories,
4. recover the true probability exactly by likelihood-ratio reweighting.

So the instanton is **not** used as a closed-form crash probability formula. It is used as a principled proposal mechanism for rare-event Monte Carlo.

---

## Forecast target

Let \(P_k\) be the daily adjusted close and \(X_k = \log P_k\). Define the running peak and drawdown on the daily grid by

- \(M_k = \max_{j \le k} X_j\),
- \(D_k = M_k - X_k \ge 0\)  (log-drawdown),
- \(\delta_k = 1 - e^{-D_k}\)  (drawdown fraction).

For a chosen drawdown fraction \(\delta \in (0,1)\), define the corresponding log-threshold

\[
d(\delta) = -\log(1-\delta).
\]

Given forecast origin day \(k\), current state \((X_k, V_k, D_k) = (x_0, v_0, d_0)\), and a horizon of \(T_{\text{days}} = n\) trading days, the quantity estimated is

\[
p_k(T_{\text{days}}, \delta)
=
\mathbb{P}\!\left(
\max_{0 \le i \le n} D_{k+i} \ge d(\delta)
\;\middle|\;
X_k = x_0,\; V_k = v_0,\; D_k = d_0
\right).
\]

### Important modelling convention

\((X,V)\) are simulated as a **2D Markov diffusion**. Drawdown \(D\) is **not** a third SDE. It is updated deterministically from the simulated \(X\)-path using the exact running-maximum recursion.

### The already-hit regime

Under the event definition above, if \(d_0 \ge d(\delta)\) then the event is already true at the forecast origin. In that case the forecast is deterministically

- \(\hat p = 1\),
- \(SE = 0\),

and no instanton / importance-sampling run is needed.

---

## Method at a glance

The implementation follows this logic:

### 1. Daily state construction
From SPY daily adjusted-close data, build a daily state table containing price, log-price, returns, running peak, drawdown, drawdown fraction, and a rolling realised-variance proxy.

### 2. Stochastic-volatility fit
Fit a 2D Markov SV model for \((X,V)\) on the same daily grid used later in simulation and optimisation.

### 3. Finite-horizon drawdown event
For each forecast date and query \((T_{\text{days}}, \delta)\), define the event that the running drawdown breaches the threshold within the forecast horizon.

### 4. Instanton computation
For each candidate hitting day \(\tau \le n\), solve a discrete minimum-action problem for a path that hits the drawdown constraint at that time. Select the minimiser over \(\tau\).

### 5. Instanton-guided proposal
Turn the selected minimum-action path into a control drift \(u^\star\), then apply a tempered/capped version in simulation to make crash-like paths much more common under a biased measure \(Q\).

### 6. Exact reweighting
Recover the true probability under the original measure \(P\) using the exact pathwise likelihood ratio for a Gaussian shift of the Brownian increments.

### 7. Forecast evaluation
Evaluate forecast quality using realised hit labels, proper scoring rules, calibration/reliability summaries, and optional naive-Monte-Carlo comparison where brute-force simulation is still feasible.

---

## Main modelling conventions

Implemented in `src/constants.py`, `src/model_sv.py`, and used consistently throughout optimisation and simulation:

- **Daily grid:** \(\Delta t = 1/252\) years.
- **Default query grid:** \(\delta \in \{0.20, 0.30\}\), \(T_{\text{days}} \in \{20, 60\}\).
- **Threshold transform:** \(d(\delta) = -\log(1-\delta)\).
- **Drawdown update:** exact running maximum in simulation/evaluation; smooth surrogate only inside the optimiser.
- **Effective variance regularisation:**  
  \[
  v_{\text{eff}} = v_{\min} + \mathrm{softplus}_{\alpha_v}(v - v_{\min}),
  \]
  used in coefficients, action, and simulation.
- **Importance-sampling control:** \(u_i = \mathrm{cap}(\lambda u_i^\star)\), with post-hit truncation.
- **Likelihood ratio:** exact Gaussian shift on independent Brownian increments.

---

## Repository layout

```text
Instanton Crash Analysis/
├─ data/
│  ├─ raw/                       # fetched SPY adjusted close etc.
│  └─ processed/                 # state table: X, r, M, D, delta, V_proxy, ...
│
├─ src/
│  ├─ constants.py               # daily grid, committed query grid, regularisation + IS knobs
│  ├─ data_state.py              # Step 1: state construction
│  ├─ fit.py                     # Step 2: QMLE-style fit for (µ, κ, v̄, ξ, ρ)
│  ├─ model_sv.py                # variance regularisation
│  ├─ instanton.py               # fixed-time MAM, τ-search, instanton -> control
│  ├─ importance_sampling.py     # IS simulation, LR, estimator, ESS
│  ├─ evaluation.py              # labels, scores, reliability, naive MC helpers
│  ├─ plotting.py                # shared plot helpers
│  └─ io.py                      # save/load helpers, metadata, instanton caching
│
├─ scripts/
│  ├─ 00_fetch_data.py
│  ├─ 01_build_state.py
│  ├─ 02_fit_model.py
│  ├─ 03_daily_forecast.py       # daily instanton + IS forecasts
│  ├─ 04_run_evaluation.py       # labels, scores, reliability and figures
│
├─ outputs/
│  ├─ params/                    # fitted parameter snapshots
│  └─ runs/                      # per-run forecast/evaluation outputs
│
├─ tests/
└─ README.md
```

---

## Demo result summary

The repository is set up around a demonstration on **SPY during the 2007–2008 crisis**.

Default evaluation window:

- **start:** 2007-01-01
- **end:** 2008-10-31

Default forecast queries:

- \(T_{\text{days}} \in \{20, 60\}\)
- \(\delta \in \{0.20, 0.30\}\)

That gives four fixed forecast definitions:
- 20 trading days, 20% drawdown
- 20 trading days, 30% drawdown
- 60 trading days, 20% drawdown
- 60 trading days, 30% drawdown

In the reported run, each date-query pair used:

- \(N = 5000\) importance-sampling paths,
- tempering \(\lambda = 0.5\),
- control cap \(u_{\max} = 10\).

### Numerical summary from the demo

| Query | Median ESS | Mean relative SE | Mean weight | ECE |
|---|---:|---:|---:|---:|
| \(T=20,\ \delta=0.20\) | 710 | 0.116 | 0.999 | 0.034 |
| \(T=20,\ \delta=0.30\) | 81 | 0.193 | 0.997 | 0.033 |
| \(T=60,\ \delta=0.20\) | 2247 | 0.072 | 1.002 | 0.113 |
| \(T=60,\ \delta=0.30\) | 601 | 0.134 | 1.007 | 0.101 |

Across the sample, the expected event-set ordering held:

\[
p_k(60, d) \ge p_k(20, d), \qquad
p_k(n, d_{0.20}) \ge p_k(n, d_{0.30}).
\]

So longer horizons and milder thresholds consistently produce larger crash probabilities.

---

## Quickstart

### Install Python dependencies
This repo is pure-Python (no compiled extensions). A minimal set of dependencies to run the full pipeline is:

```bash
python -m pip install numpy pandas scipy matplotlib yfinance
```

(For tests: `python -m pip install pytest`.)

### 0) Fetch SPY daily data
```bash
python scripts/00_fetch_data.py
```

Writes `data/raw/` (CSV + metadata JSON).

### 1) Build the state table
```bash
python scripts/01_build_state.py --ticker SPY --price_col "Adj Close" --rv_window_days 20
```

Writes `data/processed/state_*.csv` with columns such as:

- `P`
- `X`
- `r`
- `M`
- `D`
- `delta`
- `V_proxy`

### 2) Fit SV parameters
Default training cutoff is pre-crisis:

```bash
python scripts/02_fit_model.py --ticker SPY --train_end 2006-12-31
```

Writes `outputs/params/sv_params_*.json`.

### 3) Run daily forecasts
```bash
python scripts/03_daily_forecast.py
```

Notes:

- The forecasting window is currently **fixed in the script** to 2007-01-01 through 2008-10-31 (the committed demo window).
- Parallel execution (`--jobs > 1`) is intentionally guarded because block-local warm starts can change the selected instanton; use:

```bash
python scripts/03_daily_forecast.py --jobs 8 --allow_parallel_nondeterminism
```

Useful options:

- `--tdays 20`
- `--delta 0.20`
- `--n_paths 20000`
- `--seed_mode per_query`

This writes a self-contained run directory under `outputs/runs/`.

### 4) Run evaluation
```bash
python scripts/04_run_evaluation.py --run_dir outputs/runs/<your_run_folder>
```

This writes joined forecast/label tables, scores, reliability bins, and plots to:

```text
outputs/runs/<your_run_folder>/diagnostics/
```

Optional naive-vs-IS check:
```bash
python scripts/04_run_evaluation.py \
  --run_dir outputs/runs/<your_run_folder> \
  --run_naive_checks \
  --naive_n 20000 --is_n 5000
```

---

## Outputs

### Forecast CSV
Each row corresponds to one \((\text{Date}, T_{\text{days}}, \delta)\) query and includes fields such as

- `p_hat`
- `se`
- `ess`
- `mean_weight`
- `max_weight`
- `hit_rate_Q`
- `tau_star`
- `alpha_m`
- `action`
- `d` (log-threshold)
- `dtilde_tau`, `dexact_tau`, `exact_shortfall`
- `prehit_max_violation`
- `hit_err`
- `instanton_file`

### Diagnostics
`04_run_evaluation.py` produces files such as

- `eval_joined_*.csv` (forecast rows joined with realised labels)
- `eval_*.json` (summary + input fingerprints + output manifest)
- `scores_full_by_group_*.csv`, `scores_prehit_by_group_*.csv`
- `baseline_full_by_group_*.csv`, `baseline_prehit_by_group_*.csv`
- `score_comparison_full_by_group_*.csv`, `score_comparison_prehit_by_group_*.csv`
- `reliability_full_*.csv`, `reliability_prehit_*.csv`
- `calibration_full_*.csv`, `calibration_prehit_*.csv`
- `is_diagnostics_full_*.csv`, `is_diagnostics_prehit_*.csv`
- `bootstrap_scores_full_overall_*.csv`, `bootstrap_scores_full_by_group_*.csv`
- `bootstrap_scores_prehit_overall_*.csv`, `bootstrap_scores_prehit_by_group_*.csv`
- `ordering_full_*.csv`, `ordering_prehit_*.csv` (probability-ordering checks)
- `plots_*/` (default on; set `--make_plots false` to disable)
- `naive_mc_checks_*.csv` (optional; only with `--run_naive_checks`)

---

## Reproducibility

Each stage writes metadata recording:

- raw data provenance,
- state-build parameters,
- fitted parameters and diagnostics,
- forecast run configuration,
- evaluation configuration and outputs.

This is intended to make each run auditable and reproducible.

---

## Limitations and next steps

This repository demonstrates that instanton-guided importance sampling can make **rare, path-dependent drawdown probabilities** computationally accessible within a stochastic-volatility setting.

At the same time, the current framework has important limitations:

- it remains a **diffusion** model,
- it does **not yet fully incorporate jump-driven or genuinely heavy-tailed dynamics**,
- it has not yet been benchmarked in a fully matched forecasting comparison against alternatives such as Gaussian-GARCH, Student-t GARCH, or jump-augmented models.

So this project should be read as a **methodological and computational demonstration**, not as a final word on crash forecasting.

---

## Testing

```bash
pytest -q
```

---

## References

[1] Britannica Editors. *Brownian Motion*. Encyclopaedia Britannica. 2026.

[2] Dilip B. Madan. “Stochastic Processes in Finance”. In: *Annual Review of Financial Economics* 2 (2010), pp. 277–314. DOI: 10.1146/annurev.financial.050808.114506.

[3] Vladimir N. Soloviev and Yurii Romanenko. “Economic analog of Heisenberg uncertainty principle and financial crisis”. In: *System Analysis and Information Technology: 19th International Conference* (2017).

[4] Andrii Bielinskyi et al. “Econophysics of cryptocurrency crashes: an overview”. In: *arXiv preprint arXiv:2101.08136* (2021).

[5] Andrii O. Bielinskyi et al. “Predictors of oil shocks. Econophysical approach in environmental science”. In: *IOP Conference Series: Earth and Environmental Science* 628 (2021), p. 012019. DOI: 10.1088/1755-1315/628/1/012019.

[6] Hélène Rey. *Financial crises, global conditions and regime changes*. VoxEU, Centre for Economic Policy Research. 2013.

[7] U.S. Securities and Exchange Commission. *Bear Market*. Investor.gov. 2026.

[8] Benoit Mandelbrot. “The Variation of Certain Speculative Prices”. In: *The Journal of Business* 36.4 (1963), pp. 394–419. DOI: 10.1086/294632.

[9] Eugene F. Fama. “The Behavior of Stock-Market Prices”. In: *The Journal of Business* 38.1 (1965), pp. 34–105.

[10] Rama Cont. “Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues”. In: *Quantitative Finance* 1.2 (2001), pp. 223–236. DOI: 10.1080/713665670.

[11] Francois M. Longin. “The Asymptotic Distribution of Extreme Stock Market Returns”. In: *The Journal of Business* 69.3 (1996), pp. 383–408. DOI: 10.1086/209695.

[12] Alexander J. McNeil, Rüdiger Frey, and Paul Embrechts. *Quantitative Risk Management: Concepts, Techniques, and Tools*. Princeton University Press, 2005.

[13] Rosario N. Mantegna and H. Eugene Stanley. “Scaling Behaviour in the Dynamics of an Economic Index”. In: *Nature* 376 (1995), pp. 46–49. DOI: 10.1038/376046a0.

[14] Rosario N. Mantegna and H. Eugene Stanley. *Introduction to Econophysics: Correlations and Complexity in Finance*. Cambridge University Press, 1999.

[15] Jean-Philippe Bouchaud and Marc Potters. *Theory of Financial Risk and Derivative Pricing: From Statistical Physics to Risk Management*. 2nd ed. Cambridge University Press, 2003.

[16] Tim Bollerslev. “Generalized Autoregressive Conditional Heteroskedasticity”. In: *Journal of Econometrics* 31.3 (1986), pp. 307–327.

[17] H. Eugene Stanley. *Introduction to Phase Transitions and Critical Phenomena*. Oxford University Press, 1971.

[18] Zhuanxin Ding, Clive W. J. Granger, and Robert F. Engle. “A Long Memory Property of Stock Market Returns and a New Model”. In: *Journal of Empirical Finance* 1.1 (1993), pp. 83–106. DOI: 10.1016/0927-5398(93)90006-D.

[19] Daniel B. Nelson. “Conditional Heteroskedasticity in Asset Returns: A New Approach”. In: *Econometrica* 59.2 (1991), pp. 347–370.

[20] Lawrence R. Glosten, Ravi Jagannathan, and David E. Runkle. “On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks”. In: *The Journal of Finance* 48.5 (1993), pp. 1779–1801.

[21] Paul Embrechts, Claudia Klüppelberg, and Thomas Mikosch. *Modelling Extremal Events for Insurance and Finance*. Springer, 1997.

[22] Ralf Metzler and Joseph Klafter. “The Random Walk’s Guide to Anomalous Diffusion: A Fractional Dynamics Approach”. In: *Physics Reports* 339.1 (2000), pp. 1–77.

[23] Louis Bachelier. *Théorie de la spéculation*. Paris: Gauthier-Villars, 1900.

[24] Boris V. Gnedenko and Andrey N. Kolmogorov. *Limit Distributions for Sums of Independent Random Variables*. Addison-Wesley, 1954.

[25] Jon Danielsson. *Financial Risk Forecasting: The Theory and Practice of Forecasting Market Risk with Implementation in R and MATLAB*. Wiley, 2011.

[26] Christian Beck and E. G. D. Cohen. “Superstatistics”. In: *Physica A: Statistical Mechanics and its Applications* 322 (2003), pp. 267–275. DOI: 10.1016/S0378-4371(03)00019-0.

[27] Vladimir M. Zolotarev. *One-Dimensional Stable Distributions*. American Mathematical Society, 1986.

[28] Jan Rosiński. “Tempering Stable Processes”. In: *Stochastic Processes and their Applications* 117.6 (2007), pp. 677–707. DOI: 10.1016/j.spa.2006.10.003.

[29] Rosario N. Mantegna and H. Eugene Stanley. “Stochastic Process with Ultraslow Convergence to a Gaussian: The Truncated Lévy Flight”. In: *Physical Review Letters* 73.22 (1994), pp. 2946–2949. DOI: 10.1103/PhysRevLett.73.2946.

[30] Peter Carr et al. “The Fine Structure of Asset Returns: An Empirical Investigation”. In: *The Journal of Business* 75.2 (2002), pp. 305–332.

[31] G. William Schwert. “Stock Volatility and the Crash of ’87”. In: *The Review of Financial Studies* 3.1 (1990), pp. 77–102.

[32] Nassim Nicholas Taleb. *The Black Swan: The Impact of the Highly Improbable*. Random House, 2007.

[33] Didier Sornette. “Critical Market Crashes”. In: *Physics Reports* 378.1 (2003), pp. 1–98.

[34] Basel Committee on Banking Supervision. *Minimum Capital Requirements for Market Risk*. 2016.

[35] Carlo Acerbi and Dirk Tasche. “On the Coherence of Expected Shortfall”. In: *Journal of Banking & Finance* 26.7 (2002), pp. 1487–1503. DOI: 10.1016/S0378-4266(02)00283-2.

[36] Andrew Ang and Joseph Chen. “Asymmetric Correlations of Equity Portfolios”. In: *Journal of Financial Economics* 63.3 (2002), pp. 443–494.

[37] Roger B. Nelsen. *An Introduction to Copulas*. 2nd ed. Springer, 2006.

[38] Paul Embrechts, Alexander McNeil, and Daniel Straumann. “Correlation and Dependence in Risk Management: Properties and Pitfalls”. In: *Risk Management: Value at Risk and Beyond*. Ed. by M. A. H. Dempster. Cambridge University Press, 2002, pp. 176–223.

[39] Robert C. Merton. “Option Pricing when Underlying Stock Returns are Discontinuous”. In: *Journal of Financial Economics* 3.1–2 (1976), pp. 125–144.

[40] David S. Bates. “Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options”. In: *The Review of Financial Studies* 9.1 (1996), pp. 69–107. DOI: 10.1093/rfs/9.1.69.

[41] Steven L. Heston. “A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options”. In: *The Review of Financial Studies* 6.2 (1993), pp. 327–343.

[42] A. V. Chechkin et al. “Barrier Crossing Driven by Lévy Noise: Universality and the Role of Noise Intensity”. In: *Physical Review E* 75.4 (2007), p. 041101. DOI: 10.1103/PhysRevE.75.041101.

[43] Mark I. Freidlin and Alexander D. Wentzell. *Random Perturbations of Dynamical Systems*. 3rd ed. Springer, 2012.

[44] Tobias Grafke and Eric Vanden-Eijnden. “Numerical Computation of Rare Events via Large Deviation Theory”. In: *Chaos: An Interdisciplinary Journal of Nonlinear Science* 29.6 (2019), p. 063118. DOI: 10.1063/1.5084025.

[45] Sidney Coleman. *Aspects of Symmetry: Selected Erice Lectures*. Cambridge University Press, 1985.

[46] H. A. Kramers. “Brownian Motion in a Field of Force and the Diffusion Model of Chemical Reactions”. In: *Physica* 7 (1940), pp. 284–304.

[47] Sidney Coleman. *Aspects of Symmetry: Selected Erice Lectures*. Cambridge University Press, 1985.

[48] Jean Zinn-Justin. *Quantum Field Theory and Critical Phenomena*. Oxford University Press, 2002.
