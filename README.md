# Subscription Pricing Experiment — Synthetic Data & Analysis

Professional, portfolio-ready project that **simulates** a digital subscription business (streaming, SaaS, or online learning) running **randomized pricing experiments**, then demonstrates end-to-end analytics: exploratory analysis, A/B testing, revenue and LTV views, predictive models, and **causal interpretation**.

All numeric outcomes are produced with **explicit structural equations** and **vectorized NumPy** — not with generative-AI text-to-table tools.

---

## Business problem

The product monetizes through a monthly subscription. Product and growth teams run experiments that assign users to **headline prices** and **discounts**, and need to understand:

- **Conversion** to paid plans under different prices  
- **Revenue and ARPU** tradeoffs  
- **Churn and retention** after conversion  
- **Lifetime value (LTV)** under operational noise (billing failures, refunds)  
- **Segment-specific** responses (students, prior trialists, device mix, etc.)

This repository provides a **synthetic but realistic** environment to practice experiment analysis, predictive modeling, and causal reasoning when **unobserved preferences** confound simple regressions.

---

## Architecture

| Layer | Role |
|--------|------|
| `src/simulation.py` | Latent factors, treatment assignment, logistic conversion, geometric survival (churn), revenue/LTV, export-safe tables |
| `src/utils.py` | Seeds, numerics, sampling helpers |
| `src/generator.py` | CLI / programmatic CSV export + run summary JSON |
| `synthetic_data_generator.ipynb` | Walkthrough of methodology, equations, and generation |
| `pricing_experiment_analysis.ipynb` | EDA, A/B tests, revenue, ML, causal discussion |
| `output/` | `users.csv`, `pricing_experiments.csv`, `subscriptions.csv`, `generation_summary.json` |

```text
users (1 row / user)
   |
   +----< pricing_experiments (>=1 row / user; rare duplicate log rows)
   |
   +----< subscriptions (0 or 1 row / user in this sim; only subscribers)
```

---

## Table relationships

- **`users.user_id`**: primary key for the customer dimension.  
- **`pricing_experiments.user_id`**: foreign key to `users`. Grain is **one primary assignment per user** plus **optional duplicate logging rows** (same `user_id`, new `experiment_row_id`) to mimic pipeline bugs.  
- **`subscriptions.user_id`**: foreign key to `users`. Grain is **at most one subscription record per user** in this simplified simulator (first conversion only).

See **`schema.md`** for an ERD-style description and **`data_dictionary.md`** for every column.

---

## Simulation methodology (high level)

1. **User features** — demographics, device, acquisition, noisy engagement proxies.  
2. **Latent variables** — willingness to pay, price sensitivity, intrinsic engagement, brand affinity (correlated).  
3. **RCT assignment** — users randomized to `control` / `variant_mid` / `variant_high` with different list prices and discounts.  
4. **Funnel** — view pricing page → click subscribe → subscribe (Bernoulli draws at each stage).  
5. **Conversion** — logistic model in assigned price, observed covariates, and **latent terms** (omitted in exports).  
6. **Retention** — **geometric** survival (monthly churn hazard) with hazard increasing when **price exceeds latent WTP**, engagement is low, or billing fails stack up.  
7. **Revenue / LTV** — months active × paid price, minus refunds and promo credits, with right-skewed noise and rare **LTV outliers**.  
8. **Messiness** — MCAR-style missing `age` / `income_segment`, partial `churn_reason` missingness, duplicate experiment rows, version fragmentation.

---

## Omitted variables and confounding

The simulator generates **latent** constructs that directly enter the conversion and churn structural equations:

| Latent (internal only) | Influences |
|------------------------|------------|
| `willingness_to_pay` | Conversion, churn when price > WTP |
| `price_sensitivity` | Conversion (interaction with price), refunds |
| `intrinsic_engagement` | Observed engagement proxy, funnel, churn |
| `latent_brand_affinity` | Observed engagement proxy, funnel, churn |

**These columns are never written to CSV.** Analysts only see **noisy proxies** (e.g. `engagement_score`). Because treatment is **randomized**, the **average treatment effect** of price on conversion is identified under standard RCT assumptions. However, **non-experimental** comparisons (e.g. comparing high-LTV users to low-LTV users) remain vulnerable to **omitted variable bias** — matching the way real teams must separate experiment readouts from observational slices.

---

## Structural equations (implemented)

**Conversion (logistic).** With effective price $p^{eff}$ (list price adjusted for perceived discount), observed engagement $E^{obs}_i$, indicators for prior trial and student status, and latent terms:

$$
\begin{aligned}
\mathbb{P}(\text{subscribe}_i = 1)
  &= \sigma\Big(
    \beta_0
    + \beta_1 \tilde{p}^{eff}_i
    + \beta_2 \tilde{E}^{obs}_i
    + \beta_3\, \text{prior\_trial}_i
    + \beta_4\, \text{student}_i \\
  &\quad
    + \beta_{wtp}\, \frac{WTP_i - p^{eff}_i}{8}
    + \beta_{ps}\, S_i \cdot \frac{p^{eff}_i}{10}
    + \beta_{eng}\, E^{*}_i
    + \beta_{brand}\, B_i
    + \beta_{int}\, \frac{WTP_i - p^{eff}_i}{8}\, S_i
  \Big)
\end{aligned}
$$

*Plain summary:* log-odds of subscribe = linear index in price, observed engagement, trial/student dummies, latent WTP gap, price $\times$ sensitivity, intrinsic engagement, brand, and WTP-gap $\times$ sensitivity; then logistic $\sigma(\cdot)$.

Here $\sigma$ is the logistic function; $WTP_i$ is latent willingness to pay; $S_i$ price sensitivity; $E^{*}_i$ intrinsic engagement; $B_i$ brand affinity; tildes denote centering/scaling as in code.

**Churn (geometric / discrete hazard).** Each paid month, churn with probability $q_i$:

$$
q_i = \mathrm{clip}\left(
  \sigma\left(
    \gamma_0
    + \gamma_1\, \frac{p^{paid}_i - WTP_i}{5}
    + \gamma_2\, \frac{40 - E^{obs}_i}{15}
    + \gamma_3\, \text{billing\_failures}_i
    + \gamma_4\, E^{*}_i
    + \gamma_5\, (-B_i)
  \right)
\right)
$$

*Plain summary:* monthly churn probability = clipped logistic in (price minus WTP), low engagement, billing failures, intrinsic engagement, and brand.

**Survival length** (months, capped): $T_i = \min(T^{\max},\, \text{Geometric}(q_i))$ using NumPy's `numpy.random.Generator.geometric` parameterization (probability of “success” = churn event each period).

**Revenue and LTV (conceptual).**

$$
\begin{aligned}
\text{gross}_i &= T_i \cdot p^{paid}_i \\
\text{net}_i &= \text{gross}_i - \text{refund}_i - \text{promo}_i + \epsilon_i \\
LTV_i &\approx \frac{\text{net}_i}{(1+r)^{T_i/12}}
\end{aligned}
$$

*Plain summary:* gross = tenure $\times$ paid price; net subtracts refunds/promos and noise; LTV applies a small annualized discount $r$.

Full detail and code: `src/simulation.py` and `synthetic_data_generator.ipynb`.

---

## Realism features

- **Missing data** — e.g. `age`, `income_segment`, and some `churn_reason` values.  
- **Outliers** — heavy-tailed WTP in latent space; occasional extreme LTV multipliers.  
- **Skew** — right-skewed revenue / LTV; lognormal-style usage.  
- **Class imbalance** — rare refunds, imbalanced churn reasons, uneven treatment sizes (configurable).  
- **Operational noise** — billing failures, promo credits, billing anomaly flags, duplicate experiment events.

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### Generate data (scalable)

```bash
# Default: 10k users, seed 42, writes to ./output
python -m src.generator --n-rows 10000 --seed 42 --output-dir output

# Large-scale example (ensure enough RAM; ~1M users is feasible on many laptops)
python -m src.generator --n-rows 1000000 --seed 42 --output-dir output
```

Options:

- `--no-duplicates` — disable duplicate experiment log rows.  
- `--max-months` — cap simulated subscription horizon.

From Python:

```python
from src.generator import generate
generate(n_users=50_000, seed=123, output_dir="output")
```

---

## Sample outputs

After generation, `output/` contains:

- `users.csv` — one row per user.  
- `pricing_experiments.csv` — experiment exposure + funnel + subscribe outcome.  
- `subscriptions.csv` — subscribers only, with revenue and churn fields.  
- `generation_summary.json` — quick run diagnostics (conversion rate, churn share, etc.).

Example summary from a 10k run (seed **42**): overall conversion $\approx$ **15.9%**; median subscriber LTV (USD-heavy slice) $\approx$ **180 USD**; see `final_report.md` for narrative analysis.

---

## Analysis summary

Open **`pricing_experiment_analysis.ipynb`** for:

1. **EDA** — distributions, missingness, correlations, cohorts, treatment contrasts.  
2. **A/B testing** — conversion differences, uplift, bootstrap CIs, $z$ and $t$ tests via `statsmodels` / scipy.  
3. **Revenue** — ARPU-style views among converts, elasticity discussion.  
4. **Predictive models** — logistic regression, random forest, XGBoost (if installed), optional MLP; metrics include ROC-AUC, precision/recall, and calibration-style interpretation notes.

Conceptual takeaway: **randomized pricing** identifies **intent-to-treat / ATE-style** contrasts for assignments used in product decisions; **regression on observational slices** remains sensitive to latent WTP unless design or instrumentation addresses it — see causal markdown sections in the notebook.

---

## License & use

Synthetic data only — safe for portfolios, coursework, and method benchmarking. Tune `SimulationConfig` in `src/simulation.py` for alternative elasticities or segment mixes.
