# Instanton Crash Analysis

Forecast short-horizon **drawdown risk** using an **instanton-guided importance sampler** for a daily-grid stochastic-volatility (SV) diffusion model.

This repository implements the end-to-end pipeline:
- build a daily **state** from SPY prices,
- fit a **2D Markov SV model** on a matched daily discretisation,
- compute **fixed-time instantons** (minimum-action paths) for drawdown-threshold hitting events via fixed-horizon MAM,
- use the instanton to design an **importance sampler** with exact Gaussian-shift likelihood ratios,
- produce daily **crash probabilities** with **Monte Carlo uncertainty** (SE/ESS),
- evaluate forecasts with proper scoring rules and reliability diagnostics, and optionally sanity-check against naive Monte Carlo where feasible.

---

## Forecast target (what we estimate each day)

Let \(P_k\) be daily adjusted close, \(X_k=\log P_k\). Define the running peak and drawdown on the **daily grid**:

- \(M_k = \max_{j\le k} X_j\)
- \(D_k = M_k - X_k \ge 0\)  (log-drawdown)
- drawdown fraction: \(\delta_k = 1-e^{-D_k}\)

For a chosen drawdown fraction \(\delta \in (0,1)\), set the **log threshold**

\[
 d(\delta) = -\log(1-\delta).
\]

Given a forecast origin day \(k\) with observed \((X_k,V_k,D_k)=(x_0,v_0,d_0)\) and a horizon of \(T_{\text{days}}=n\) trading days, the forecast is:

\[
 p_k(T_{\text{days}},\delta)
 = \mathbb{P}\!\left(\max_{0\le i \le n} D_{k+i} \ge d(\delta)\ \middle|\ X_k=x_0, V_k=v_0, D_k=d_0\right).
\]

**Important:** \((X,V)\) are simulated as a **2D Markov diffusion**. Drawdown \(D\) is *not* a third SDE; it is updated deterministically from the simulated \(X\) path using the **exact running-maximum recursion**.

### The \(\tau^*=0\) (already-hit) regime

Under the definition \(\max_{0\le i\le n}D_{k+i}\ge d\) (origin included), if \(d_0 \ge d(\delta)\) then the event is already true at the forecast origin and the model returns:
- \(\hat p = 1\),
- \(SE = 0\),
- no instanton/IS simulation is needed for that query.

This is expected and consistent with the framework.

---

## Repository layout

```
Instanton Crash Analysis/
├─ data/
│  ├─ raw/                       # fetched SPY adjusted close etc.
│  └─ processed/                 # state table: X, r, M, D, delta, V_proxy, ...
│
├─ src/
│  ├─ constants.py               # daily grid, committed query grid, regularisation + IS knobs
│  ├─ data_state.py              # Step 1: state construction
│  ├─ fit.py                     # Step 2: QMLE-style fit for (µ, κ, v̄, ξ, ρ)
│  ├─ model_sv.py                # v_eff = vmin + softplus_{αv}(v - vmin)
│  ├─ instanton.py               # Step 4: fixed-time MAM, τ search, instanton -> control u*
│  ├─ importance_sampling.py     # Step 5: IS simulation, LR, estimator, ESS
│  ├─ evaluation.py              # Step 7: labels, scores, reliability, naive MC helpers
│  ├─ plotting.py                # shared plot helpers
│  └─ io.py                      # save/load helpers, run metadata, instanton caching helpers
│
├─ scripts/
│  ├─ 00_fetch_data.py
│  ├─ 01_build_state.py
│  ├─ 02_fit_model.py
│  ├─ 03_daily_forecast.py       # daily instanton+IS forecasts (writes a run folder)
│  ├─ 04_run_evaluation.py       # join labels, scores, reliability, optional naive-vs-IS checks
│  └─ 05_make_figures.py         # generate publication-style figures
│
├─ outputs/
│  ├─ params/                    # fitted parameter snapshots (JSON)
│  └─ runs/                      # per-run outputs from scripts/03 & 04
│     └─ run_.../
│        ├─ forecasts/           # forecasts_*.csv + metadata
│        ├─ instantons/          # cached instanton .npz files
│        └─ diagnostics/         # evaluation tables, reliability, plots, naive checks
│
├─ tests/
└─ README.md
```

---

## Key modelling conventions (PDF-aligned)

Implemented in `src/constants.py`, `src/model_sv.py`, and enforced throughout simulation/instanton/IS:

- **Daily grid:** \(\Delta t = 1/252\) years.
- **Committed query grid:** \(\delta \in \{0.20, 0.30\}\), \(T_{\text{days}} \in \{20, 60\}\) (overrideable via CLI).
- **Threshold transform:** \(d(\delta) = -\log(1-\delta)\).
- **Drawdown recursion:** exact running maximum in all simulation/evaluation; smooth surrogate only inside the optimiser.
- **Effective variance regularisation:** \(v_{\text{eff}} = v_{\min} + \mathrm{softplus}_{\alpha_v}(v-v_{\min})\) used **everywhere** in coefficients/action/simulation.
- **Importance sampling control:** \(u_i = \mathrm{cap}(\lambda u^*_i)\), with \(u^*_i = 0\) for \(i\ge \tau^*\).
- **Likelihood ratio:** exact Gaussian shift on independent Brownian increments.

---

## Setup

Create and activate a virtual environment, then install dependencies (choose one):

**Option A: editable install (recommended)**
```bash
pip install -e .
```

**Option B: requirements**
```bash
pip install -r requirements.txt
```

---

## Quickstart (end-to-end)

### 0) Fetch SPY daily data
```bash
python scripts/00_fetch_data.py
```
Writes `data/raw/` (CSV + metadata JSON).

### 1) Build the state table (Step 1)
```bash
python scripts/01_build_state.py --ticker SPY --price_col "Adj Close" --rv_window_days 20
```
Writes `data/processed/state_*.csv` with columns: `P, X, r, M, D, delta, V_proxy`.

### 2) Fit SV parameters (Step 2)
Default training ends 2006-12-31 (pre-episode fit):
```bash
python scripts/02_fit_model.py --ticker SPY --train_end 2006-12-31
```
Writes `outputs/params/sv_params_*.json`.

### 3) Daily forecasts (instantons + IS) (Steps 3–6)

`03_daily_forecast.py` writes a **self-contained run directory** under `outputs/runs/` containing forecasts + cached instantons.

By default, forecasts are produced over:
- start: **2007-01-01**
- end: **2008-10-31**

```bash
python scripts/03_daily_forecast.py --jobs 8
```

Useful options:
- `--tdays 20` or `--delta 0.20` to run a single query
- `--n_paths 20000` to reduce SE (costly)
- `--seed_mode per_query` for stable results across parallel chunking

### 4) Evaluation and reliability (Step 7)

Run evaluation directly on the run directory:

```bash
python scripts/04_run_evaluation.py --run_dir outputs/runs/<your_run_folder>
```

This writes joined forecast/label tables, scores, reliability bins, and plots into:
`outputs/runs/<run_folder>/diagnostics/`

Optional: sanity-check IS against naive MC on feasible rows:
```bash
python scripts/04_run_evaluation.py \
  --run_dir outputs/runs/<your_run_folder> \
  --run_naive_checks \
  --naive_n 20000 --is_n 5000
```

### 5) Make figures

If you want the figure set, point the script at the run’s folders:

```bash
python scripts/05_make_figures.py \
  --forecasts_dir outputs/runs/<your_run_folder>/forecasts \
  --diagnostics_dir outputs/runs/<your_run_folder>/diagnostics \
  --instantons_dir outputs/runs/<your_run_folder>/instantons \
  --fig_dir outputs/runs/<your_run_folder>/diagnostics/figures
```

---

## Outputs (what to look at)

### Forecast CSV (per run)
Each row corresponds to one \((\text{Date}, T_{\text{days}}, \delta)\) query and includes:
- `p_hat`, `se`, `ess`, `mean_weight`, `max_weight`, `hit_rate_Q`
- instanton diagnostics: `tau_star`, `action`, `prehit_max_violation`, `hit_err`
- `instanton_file` (cached `.npz` used to reconstruct the IS control)

### Diagnostics and evaluation
`04_run_evaluation.py` produces:
- `eval_joined_*.csv`: forecasts + realised labels (`y`) and diagnostics
- `scores_by_group_*.csv`: mean log score + mean Brier by (Tdays, δ)
- `reliability_grouped_*.csv`: binned calibration tables (by group)
- optional `naive_mc_checks_*.csv`: naive MC vs IS reruns + z-scores

---

## Reproducibility

Each stage writes metadata (JSON) recording:
- raw data provenance,
- state-build parameters,
- fitted parameters + diagnostics,
- forecast run configuration (grid, λ, umax, seeds, solver knobs),
- evaluation configuration and outputs.

---

## Testing

```bash
pytest -q
```

---

## References
