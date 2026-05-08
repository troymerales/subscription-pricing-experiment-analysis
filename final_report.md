# Final analytical report — synthetic subscription pricing experiment

**Audience:** Growth, product analytics, data science reviewers  
**Dataset:** Synthetic CSV outputs in `output/` (generated with `--seed 42`, `n_users=10_000`).  
**Method:** Vectorized structural simulation with **latent** willingness to pay / sensitivity constructs **stripped before export**.

---

## 1. Executive summary

This cohort simulates a digital subscription funnel with **three randomly assigned pricing arms**:

- **`control`** — list price \$9.99, 0% discount  
- **`variant_mid`** — \$12.99, 10% discount  
- **`variant_high`** — \$14.99, 15% discount  

Across the simulated run (**10,000** unique users enrolled, **10,005** experiment log rows after intentional duplicates):

| Metric (person-level experiment table, duplicates removed) | Value |
|---|---:|
| Overall **conversion rate** (\`subscribed=1`) | **15.94%** |
| Conversions **`control`** | **16.89%** |
| Conversions **`variant_mid`** | **16.45%** |
| Conversions **`variant_high`** | **14.46%** |
| Subscribers with subscription rows (`subscriptions.csv`) | **1,594** |
| Share of subscribers with **post-conversion churn** (`churned=1`) | **~84.3%** (includes horizon-censored churn labels in sim) |
| Median subscriber **months active** | **17** paid months |
| Median approximate **USD LTV among USD-labeled subscriptions** | **\$179.68** |
| Subscriber **refund_flag** prevalence | **~7.0%** |

Higher posted prices mechanically depress modeled conversion versus `control`; discounts partially offset sticker shock in the latent conversion equation. Retention stays price-sensitive relative to latent WTP exported users never observe, yielding **skewed revenue and LTV** with rare super-users.

See **`pricing_experiment_analysis.ipynb`** for formal two-proportion **z-tests**, **bootstrap** percentile intervals, **Welch t-tests**, **Holm**-adjusted multiplicity, **ARPU**, and **classification** benchmarks (logistic regression, random forest, XGBoost optional, neural net).

---

## 2. Data quality and realism

Designed artifacts mirror production analytics workloads:

- **Missingness:** `age` ~5–6%; `income_segment` ~8–9%; `churn_reason` selectively missing among churn events.  
- **Outliers & skew:** Revenue / LTV are right-skew; a configurable fraction receives **elevated longevity + LTV** multipliers.  
- **Operational noise:** `billing_failures`, `billing_anomaly_flag`, stochastic refunds, fragmented `app_version`.  
- **Duplicates:** fractional duplicate rows in event-level `pricing_experiments`; analyst must **collapse to person grain** (`sort_values('experiment_row_id').drop_duplicates('user_id','first')`).  
- **Class imbalance:** low refund rate vs non-refund; imbalanced churn reason categories among churners.

---

## 3. Causal framing (experiment vs observational slicing)

Pricing tests are causal because **assigned treatment** \(Z\) precedes downstream outcomes \(Y\) and is **orthogonal to latent constructs in expectation**:

\[
\mathbb{E}[ Y_i \mid Z=z ] \text{ contrasts are identified for ITT contrasts under RCT assumptions.}
\]

However, regressions pooling users without referencing the experiment carve out **risk omitted-variable bias**:

- **`engagement_score` is only a noisy proxy for intrinsic motivation** correlated with latent WTP → conditioning naïvely is not a substitute for causal design.  
- **Simple price–quantity regressions on observational data** confound heterogeneous segment mix unless identified by instrument/experiment/design.

Recommendation: prioritize **experiment-based readouts for pricing decisions**, use predictive models **for prioritization**, and treat observational regressions **as exploratory** absent stronger identification strategy.

---

## 4. Revenue and elasticity narratives

Interpret **elasticity cautiously**:

- Assigned price affects **conversion** (extensive margin) **and** modeled **paid retention** relative to latent WTP (intensive margin).  
- **Intent-to-treat ARPU** (assign revenue zeros to non-converters before averaging) aligns with causal assignment; **subscriber-only averages** emphasize conditional pricing but lose randomization semantics.

Plots and tables are in **`pricing_experiment_analysis.ipynb`**.

---

## 5. Predictive modeling takeaway

Across conversion and churn:

- **Logistic regression** supplies interpretable directional signs (with preprocessing + class weights).  
- **Random forests / boosted trees / MLP** may increase **ROC-AUC** useful for uplift-style ranking.  
- **Latent constructs remain unobservable** ⇒ models cannot faithfully recover causal price elasticities absent explicit experiment structure.

Evaluate models with **ROC-AUC**, precision/recall, calibration checks (recommended extension), and stakeholder preference for transparency vs uplift performance.

---

## 6. Reproducibility

```bash
python -m pip install -r requirements.txt
python -m src.generator --n-rows 10000 --seed 42 --output-dir output
```

Open `pricing_experiment_analysis.ipynb` afterward to refresh visuals and statistics. Figures and tables change if `--seed` or structural coefficients (`SimulationConfig` in `src/simulation.py`) are edited.

---

## 7. Artifacts checklist

| File | Purpose |
|------|---------|
| `README.md` | Orientation, equations, causal discussion |
| `schema.md` | ERD grain + keys |
| `data_dictionary.md` | Column-level documentation + latent appendix |
| `src/simulation.py` | Structural simulator |
| `src/generator.py` | CSV export + summary JSON |
| `synthetic_data_generator.ipynb` | Methodology walkthrough |
| `pricing_experiment_analysis.ipynb` | Full portfolio analysis suite |
| `output/*.csv` | Analysis-ready relational extracts |
| `output/generation_summary.json` | Run-level KPI snapshot |
