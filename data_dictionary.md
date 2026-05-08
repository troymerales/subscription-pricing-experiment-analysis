# Data dictionary

All **exported** tables are written to `output/*.csv`. **Latent** variables are documented for transparency but **never** appear in CSV files.

Convention: `included_in_final_export` is **yes** for shipped columns, **no** for internal latents.

---

## Latent / omitted variables (simulation only)

| column_name | table_name | dtype | description | allowed values / ranges | generated_from | included_in_final_export |
|-------------|-----------|-------|-------------|-------------------------|----------------|--------------------------|
| willingness_to_pay | *(internal)* | float64 | Latent maximum monthly price user would tolerate before sharply reducing demand. | Roughly [3, 80] USD-equivalent before churn/conversion scaling | Log-normal transform of Gaussian copula factor | no |
| price_sensitivity | *(internal)* | float64 | Higher values imply stronger negative response to price increases. | Approx. [-3.5, 3.5] standardized | Gaussian copula factor, clipped | no |
| intrinsic_engagement | *(internal)* | float64 | True long-run engagement; drives observed engagement with noise. | Approx. [-3.5, 3.5] standardized | Gaussian copula factor, clipped | no |
| latent_brand_affinity | *(internal)* | float64 | Unobserved loyalty toward the product / brand. | Approx. [-3.5, 3.5] standardized | Gaussian copula factor, clipped | no |

---

## Table: `users`

| column_name | dtype | description | allowed values / ranges | generated_from | included_in_final_export |
|-------------|-------|-------------|-------------------------|----------------|--------------------------|
| user_id | int64 | Surrogate primary key. | 1 … N | Sequential | yes |
| signup_date | string (ISO date) | Calendar date of registration. | Between `study_start` and `study_end` config | Uniform offset from study window | yes |
| age | float64 | Age in years; may be missing. | [16, 78] when present; else NaN | Normal + student shift; MCAR masked | yes |
| country | string | Country bucket (marketing region). | US, PH, IN, GB, DE, BR, SG, AU, CA, NG | Weighted categorical | yes |
| device_type | string | Primary device class. | mobile_ios, mobile_android, web_desktop, web_mobile, tablet | Weighted categorical | yes |
| traffic_source | string | Acquisition channel label. | organic_search, paid_social, direct, referral, email, affiliate | Weighted categorical | yes |
| income_segment | string | Declared / inferred income band; may be missing. | unknown, &lt;40k, 40k_70k, 70k_100k, 100k_150k, 150k_plus; or NaN | Weighted categorical + MCAR | yes |
| engagement_score | float64 | Product engagement index (noisy proxy of intrinsic engagement). | [5, 99] | Linear in latents + Gaussian noise | yes |
| historical_platform_usage | float64 | Heavy-tailed usage index (pre-experiment). | (0.1, 500] | Lognormal with mean depending on intrinsic engagement | yes |
| email_verified | int8 | Email verification flag. | 0 / 1 | Bernoulli | yes |
| payment_method | string | Default / saved payment family. | card, paypal, apple_pay, google_pay, bank_debit | Weighted categorical | yes |
| prior_trial_user | int8 | Previously took a free trial. | 0 / 1 | Bernoulli | yes |
| student_flag | int8 | Student / edu program indicator. | 0 / 1 | Bernoulli | yes |
| timezone | string | User-reported or inferred TZ label. | UTC-08:00, UTC-05:00, UTC+00:00, UTC+01:00, UTC+05:30, UTC+08:00 | Weighted categorical | yes |
| app_version | string | Client build at signup. | 3.9.0 … 4.0.0-beta | Weighted categorical (version fragmentation) | yes |
| signup_hour_utc | int16 | Hour-of-day bucket (UTC). | 0–23 | Discrete uniform | yes |
| notification_opt_in | int8 | Marketing / product notification opt-in. | 0 / 1 | Bernoulli | yes |
| account_completeness_score | float64 | Profile completeness composite. | [0.35, 1.0] | Uniform | yes |
| cohort_week | int16 | Weeks since study start at signup. | ≥ 0 | Derived from signup_date | yes |

---

## Table: `pricing_experiments`

| column_name | dtype | description | allowed values / ranges | generated_from | included_in_final_export |
|-------------|-------|-------------|-------------------------|----------------|--------------------------|
| experiment_row_id | int64 | Unique log-row id (PK). | Strictly increasing within a run | Sequential + append for dupes | yes |
| experiment_id | int64 | Experiment identifier. | Default 1 | Config constant | yes |
| user_id | int64 | Foreign key to users. | 1 … N | Same as users | yes |
| treatment_group | string | Randomized pricing arm. | control, variant_mid, variant_high | Stratified multinomial RCT | yes |
| assigned_price | float64 | List monthly price shown (USD). | 9.99, 12.99, 14.99 in default config | Deterministic by arm | yes |
| discount_pct | float64 | Displayed / effective percent discount. | 0, 10, 15 in default | Deterministic by arm | yes |
| experiment_date | string (ISO date) | Date of assignment / exposure. | Near signup_date | signup_date + small offset | yes |
| viewed_pricing_page | int8 | Saw paywall / pricing screen. | 0 / 1 | Bernoulli with logistic latent + engagement | yes |
| clicked_subscribe | int8 | Clicked primary CTA. | 0 / 1 | Bernoulli gated on view | yes |
| subscribed | int8 | Converted to paid (final outcome). | 0 / 1 | Bernoulli from structural logit + funnel gates | yes |
| time_on_pricing_page_sec | float64 | Dwell time (noisy long tail). | [5, 600000] clipped | Lognormal | yes |
| pricing_page_scroll_depth_pct | float64 | Estimated scroll depth. | [0, 100] | Normal conditioned on viewed | yes |
| hesitate_event_count | int16 | Micro-hesitation events (rage clicks / compare plans). | Non-negative Poisson | Poisson intensity grows with price | yes |
| paywall_modal_shown | int8 | Modal presentation flag. | 0 / 1 | Bernoulli | yes |
| experiment_assignment_delay_sec | int64 | Telemetry delay seconds (artifact). | 0–89 + small bump on dupes | Uniform / shifted on dup rows | yes |

---

## Table: `subscriptions`

| column_name | dtype | description | allowed values / ranges | generated_from | included_in_final_export |
|-------------|-------|-------------|-------------------------|----------------|--------------------------|
| subscription_id | int64 | Subscriber row id (PK). | 1 … n_subscribers | Sequential | yes |
| user_id | int64 | Parent user id. | Subset of users | Subscribed users only | yes |
| experiment_id | int64 | Linking experiment id. | Default 1 | Config | yes |
| subscription_start_date | string (ISO date) | First paid period start (aligned with signup in sim). | Dates in study horizon | Derived from signup | yes |
| subscription_end_date | string / empty | End date if churned. | ISO date if churned else empty / NA | Derived from geometric survival | yes |
| months_active | int32 | Months paid (bucketed simulator clock). | 1 … max_subscription_months (+ small extension for outliers) | Geometric(min capped) + outlier shift | yes |
| monthly_price_paid | float64 | Realized paid price after discount + micro noise. | Positive float | Assigned price × discount + noise | yes |
| churned | int8 | Churn before horizon vs censored active. | 0 / 1 | Derived from capped survival | yes |
| churn_reason | string | Self-reported or inferred churn reason when churned. | price_too_high, did_not_use_enough, competing_service, billing_issue, life_event, seasonal_pause, unknown; NaN when not churned or MCAR masked | Multinomial + masking | yes |
| refund_flag | int8 | Any refund logged. | 0 / 1 | Bernoulli with logistic in tenure + latent sensitivity | yes |
| billing_failures | int64 | Count of failed charge attempts across life (simplified). | ≥ 0 | Binomial draws | yes |
| billing_anomaly_flag | int8 | Operational anomaly (reconciliation flag). | 0 / 1 | Rare Bernoulli | yes |
| total_revenue | float64 | Net realized revenue USD-ish (noise, refunds). | Real; skewed right | Months × price − refunds − promos + noise | yes |
| lifetime_value | float64 | Discounted heuristic LTV proxy. | Real; skewed right | revenue / discount factor^(months/12); outlier inflate | yes |
| plan_tier | string | Plan label. | standard, family, student | Weighted categorical | yes |
| currency | string | Billing currency marker. | USD, EUR | Mostly USD | yes |
| last_renewal_outcome | string | Last renewal telemetry bucket. | success, failed_then_recovered, failed | Weighted categorical | yes |
