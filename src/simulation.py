"""Vectorized synthetic subscription pricing experiment simulation.

Structural equations drive conversion (logistic), retention (geometric survival),
and revenue. Latent constructs are NEVER exported — only used inside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from numpy.random import Generator

from . import utils


@dataclass
class SimulationConfig:
    """Configuration for scalable simulation."""

    n_users: int = 10_000
    random_seed: int = 42
    study_start: str = "2024-01-01"
    study_end: str = "2025-12-31"
    experiment_id: int = 1
    max_subscription_months: int = 60
    include_duplicate_experiment_rows: bool = True
    duplicate_frac: float = 0.0005
    outlier_ltv_frac: float = 0.001

    # Logistic conversion coefficients (structural equation)
    beta_intercept: float = 0.8
    beta_price: float = -0.32
    beta_engagement_obs: float = 1.85
    beta_prior_trial: float = 0.55
    beta_student: float = 0.35
    # Latent-channel terms (induce confounding with observed proxies)
    beta_wtp_residual: float = 1.45
    beta_price_sensitivity: float = -1.05
    beta_intrinsic_engagement: float = 1.1
    beta_brand: float = 0.75
    beta_wtp_price_interaction: float = 0.12

    # Churn / survival (geometric parameterization per user)
    gamma_intercept: float = -2.35
    gamma_price_wtp_gap: float = 0.38
    gamma_low_engagement: float = 0.55
    gamma_billing_fail: float = 0.22
    gamma_intrinsic: float = -0.9
    gamma_brand: float = -0.35

    # Minimum churn probability floor (avoid zero)
    churn_prob_floor: float = 0.02
    churn_prob_cap: float = 0.45


@dataclass
class LatentState:
    """Internal-only latent draws and indices (not exported)."""

    willingness_to_pay: np.ndarray = field(repr=False)
    price_sensitivity: np.ndarray = field(repr=False)
    intrinsic_engagement: np.ndarray = field(repr=False)
    latent_brand_affinity: np.ndarray = field(repr=False)


def _rng(config: SimulationConfig) -> Generator:
    return np.random.default_rng(config.random_seed)


def generate_latent(n: int, rng: Generator) -> LatentState:
    """Correlated latent factors via Gaussian copula + transforms.

    - willingness_to_pay: USD-equivalent latent monthly willingness (heavy right tail).
    - price_sensitivity: standardized, higher = more elastic.
    - intrinsic_engagement: standardized true engagement driver.
    - latent_brand_affinity: standardized loyalty / brand preference.
    """
    mean = np.zeros(4)
    cov = np.array(
        [
            [1.0, 0.35, 0.28, 0.55],
            [0.35, 1.0, 0.15, 0.10],
            [0.28, 0.15, 1.0, 0.25],
            [0.55, 0.10, 0.25, 1.0],
        ]
    )
    z = rng.multivariate_normal(mean, cov, size=n)
    wtp = np.clip(np.exp(2.2 + 0.35 * z[:, 0]), 3.0, 80.0)
    price_sens = np.clip(z[:, 1], -3.5, 3.5)
    intrinsic = np.clip(z[:, 2], -3.5, 3.5)
    brand = np.clip(z[:, 3], -3.5, 3.5)
    return LatentState(
        willingness_to_pay=wtp,
        price_sensitivity=price_sens,
        intrinsic_engagement=intrinsic,
        latent_brand_affinity=brand,
    )


def assign_treatment(n: int, rng: Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized pricing arms (RCT). Returns treatment_group, assigned_price, discount_pct."""
    groups = np.asarray(["control", "variant_mid", "variant_high"])
    prices = {"control": 9.99, "variant_mid": 12.99, "variant_high": 14.99}
    discounts = {"control": 0.0, "variant_mid": 10.0, "variant_high": 15.0}
    arms = rng.choice(3, size=n, p=np.array([0.34, 0.33, 0.33]))
    grp = groups[arms]
    ap = np.array([prices[g] for g in grp], dtype=np.float64)
    dp = np.array([discounts[g] for g in grp], dtype=np.float64)
    return grp, ap, dp


def build_conversion_logit(
    config: SimulationConfig,
    latent: LatentState,
    assigned_price: np.ndarray,
    discount_pct: np.ndarray,
    engagement_score_obs: np.ndarray,
    prior_trial_user: np.ndarray,
    student_flag: np.ndarray,
) -> np.ndarray:
    """Logistic linear predictor including latent terms omitted from exported data.

    Structural form (conceptual):
        η_i = β0 + β1 * price_eff_i + β2 * engagement_obs_i + β3 * trial_i
              + β4 * student_i
              + β_wtp * (wtp_i - price_eff_i) / scale  # resale of WTP residual
              + β_ps * sensitivity_i * price_eff_i
              + β_eng * intrinsic_engagement_i (not observed in export)
              + β_brand * brand_i (not observed in export)

    Effective displayed price incorporates discount expectation (users anchor on headline price).
    """
    price_eff = assigned_price * (1.0 - (discount_pct / 100.0) * 0.65)
    wtp_gap = (latent.willingness_to_pay - price_eff) / 8.0
    eta = (
        config.beta_intercept
        + config.beta_price * (price_eff - 10.0) / 3.0
        + config.beta_engagement_obs * (engagement_score_obs - 55.0) / 20.0
        + config.beta_prior_trial * prior_trial_user
        + config.beta_student * student_flag
        + config.beta_wtp_residual * wtp_gap
        + config.beta_price_sensitivity * latent.price_sensitivity * (price_eff / 10.0)
        + config.beta_intrinsic_engagement * latent.intrinsic_engagement
        + config.beta_brand * latent.latent_brand_affinity
        + config.beta_wtp_price_interaction * wtp_gap * latent.price_sensitivity
    )
    return eta


def build_churn_probability(
    config: SimulationConfig,
    willingness_to_pay: np.ndarray,
    intrinsic_engagement: np.ndarray,
    latent_brand_affinity: np.ndarray,
    monthly_price_paid: np.ndarray,
    engagement_score_obs: np.ndarray,
    billing_failures: np.ndarray,
) -> np.ndarray:
    """Monthly churn probability after at least one paid month (geometric trials)."""
    gap = (monthly_price_paid - willingness_to_pay) / 5.0
    low_eng = np.clip((40.0 - engagement_score_obs) / 15.0, -2.0, 3.0)
    eta = (
        config.gamma_intercept
        + config.gamma_price_wtp_gap * gap
        + config.gamma_low_engagement * low_eng
        + config.gamma_billing_fail * np.clip(billing_failures, 0, 8)
        + config.gamma_intrinsic * intrinsic_engagement
        + config.gamma_brand * (-latent_brand_affinity)
    )
    p = utils.sigmoid(eta)
    return np.clip(p, config.churn_prob_floor, config.churn_prob_cap)


def simulate_users(
    n: int,
    rng: Generator,
    latent: LatentState,
    study_start: pd.Timestamp,
    study_end: pd.Timestamp,
) -> pd.DataFrame:
    """Generate user master table (observed + noise; latents not included)."""
    user_id = np.arange(1, n + 1, dtype=np.int64)
    span = (study_end - study_start).days
    signup_offset = rng.integers(0, max(span, 1), size=n)
    signup_raw = study_start + pd.to_timedelta(signup_offset.astype(np.int64), unit="D")
    signup_date = pd.Series(pd.to_datetime(signup_raw), name="signup_date")

    countries = ["US", "PH", "IN", "GB", "DE", "BR", "SG", "AU", "CA", "NG"]
    c_probs = np.array([0.28, 0.12, 0.09, 0.08, 0.06, 0.07, 0.05, 0.09, 0.10, 0.06])
    country = utils.weighted_choice(rng, countries, c_probs.tolist(), n)

    device_type = utils.weighted_choice(
        rng,
        ["mobile_ios", "mobile_android", "web_desktop", "web_mobile", "tablet"],
        [0.32, 0.28, 0.22, 0.12, 0.06],
        n,
    )
    traffic_source = utils.weighted_choice(
        rng,
        ["organic_search", "paid_social", "direct", "referral", "email", "affiliate"],
        [0.24, 0.18, 0.26, 0.12, 0.14, 0.06],
        n,
    )
    income_segment = utils.weighted_choice(
        rng,
        ["unknown", "<40k", "40k_70k", "70k_100k", "100k_150k", "150k_plus"],
        [0.05, 0.18, 0.26, 0.22, 0.17, 0.12],
        n,
    )
    timezone = utils.weighted_choice(
        rng,
        ["UTC-08:00", "UTC-05:00", "UTC+00:00", "UTC+01:00", "UTC+05:30", "UTC+08:00"],
        [0.18, 0.22, 0.12, 0.10, 0.20, 0.18],
        n,
    )
    app_version = utils.weighted_choice(
        rng,
        ["3.9.0", "3.9.1", "3.10.0", "3.10.2", "3.11.0", "3.11.1", "4.0.0-beta"],
        [0.08, 0.12, 0.20, 0.25, 0.18, 0.12, 0.05],
        n,
    )
    payment_method = utils.weighted_choice(
        rng,
        ["card", "paypal", "apple_pay", "google_pay", "bank_debit"],
        [0.52, 0.18, 0.12, 0.10, 0.08],
        n,
    )

    prior_trial_user = (rng.random(n) < 0.22).astype(np.int8)
    student_flag = (rng.random(n) < 0.14).astype(np.int8)

    # Age from mixture (skewed young for digital subs)
    age = np.round(
        np.clip(
            rng.normal(32.0, 11.0, size=n)
            * (0.85 + 0.15 * (country == "US"))
            + 3.0 * student_flag.astype(np.float64),
            16,
            78,
        )
    ).astype(np.int16)
    email_verified = (rng.random(n) < 0.91).astype(np.int8)

    # Observed engagement is noisy proxy of intrinsic_engagement + marketing / seasonality
    engagement_score = np.clip(
        48.0
        + 11.0 * latent.intrinsic_engagement
        + 4.0 * latent.latent_brand_affinity
        + rng.normal(0.0, 8.5, size=n)
        + 5.0 * student_flag
        - 3.0 * (country == "IN"),
        5.0,
        99.0,
    )
    historical_platform_usage = np.clip(
        rng.lognormal(mean=2.2 + 0.18 * latent.intrinsic_engagement, sigma=0.55, size=n),
        0.1,
        500.0,
    )

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "signup_date": signup_date,
            "age": age,
            "country": country,
            "device_type": device_type,
            "traffic_source": traffic_source,
            "income_segment": income_segment,
            "engagement_score": np.round(engagement_score, 2),
            "historical_platform_usage": np.round(historical_platform_usage, 3),
            "email_verified": email_verified,
            "payment_method": payment_method,
            "prior_trial_user": prior_trial_user,
            "student_flag": student_flag,
            "timezone": timezone,
            "app_version": app_version,
            "signup_hour_utc": rng.integers(0, 24, size=n).astype(np.int16),
            "notification_opt_in": (rng.random(n) < 0.67).astype(np.int8),
            "account_completeness_score": np.round(rng.uniform(0.35, 1.0, size=n), 2),
            "cohort_week": ((signup_date - study_start).dt.days.to_numpy() // 7).astype(np.int16),
        }
    )
    return df


def simulate_pricing_experiments(
    config: SimulationConfig,
    users: pd.DataFrame,
    rng: Generator,
    latent: LatentState,
    treatment_group: np.ndarray,
    assigned_price: np.ndarray,
    discount_pct: np.ndarray,
) -> pd.DataFrame:
    n = len(users)
    user_id = users["user_id"].to_numpy()
    study_start = pd.Timestamp(config.study_start)
    experiment_date = users["signup_date"] + pd.to_timedelta(
        rng.integers(0, 21, size=n), unit="D"
    )

    engagement = users["engagement_score"].to_numpy(dtype=np.float64)
    prior_trial = users["prior_trial_user"].to_numpy(dtype=np.float64)
    student = users["student_flag"].to_numpy(dtype=np.float64)

    eta_sub = build_conversion_logit(
        config, latent, assigned_price, discount_pct, engagement, prior_trial, student
    )
    p_sub = utils.clip_prob(utils.sigmoid(eta_sub))
    subscribed = (rng.random(n) < p_sub).astype(np.int8)

    # Funnel realism: view -> click -> subscribe (vectorized logistic steps)
    p_view = utils.clip_prob(
        utils.sigmoid(
            0.15 * (engagement - 50.0) / 15.0
            + 0.42 * latent.intrinsic_engagement
            + 0.28 * latent.latent_brand_affinity
            + rng.normal(0.0, 0.22, size=n)
        )
    )
    viewed_pricing_page = (rng.random(n) < p_view).astype(np.int8)
    p_click = utils.clip_prob(
        utils.sigmoid(
            np.where(
                viewed_pricing_page.astype(bool),
                0.22 * (engagement - 48.0) / 12.0 + 0.06 * (latent.willingness_to_pay / 15.0),
                -5.5,
            )
            + rng.normal(0.0, 0.28, size=n)
        )
    )
    clicked_subscribe = ((rng.random(n) < p_click) & viewed_pricing_page.astype(bool)).astype(np.int8)
    subscribed = np.where(clicked_subscribe == 0, 0, subscribed).astype(np.int8)

    time_on_page_s = np.clip(rng.lognormal(mean=4.6, sigma=1.05, size=n), 5.0, 600_000.0)
    hesitate_events = rng.poisson(np.clip(utils.sigmoid((assigned_price - 10) / 3.6) * 3.8, 0.2, 8.5))

    rows = pd.DataFrame(
        {
            "experiment_row_id": np.arange(1, n + 1, dtype=np.int64),
            "experiment_id": config.experiment_id,
            "user_id": user_id,
            "treatment_group": treatment_group,
            "assigned_price": np.round(assigned_price, 2),
            "discount_pct": np.round(discount_pct, 1),
            "experiment_date": experiment_date.dt.strftime("%Y-%m-%d"),
            "viewed_pricing_page": viewed_pricing_page.astype(np.int8),
            "clicked_subscribe": clicked_subscribe,
            "subscribed": subscribed,
            "time_on_pricing_page_sec": np.round(time_on_page_s, 2),
            "pricing_page_scroll_depth_pct": np.clip(
                rng.normal(
                    np.where(viewed_pricing_page, 62.0, 10.0),
                    18.0,
                    size=n,
                ),
                0,
                100,
            ).round(1),
            "hesitate_event_count": hesitate_events.astype(np.int16),
            "paywall_modal_shown": (rng.random(n) < 0.78).astype(np.int8),
            "experiment_assignment_delay_sec": rng.integers(0, 90, size=n),
        }
    )

    if config.include_duplicate_experiment_rows and config.duplicate_frac > 0:
        dup_n = max(1, int(n * config.duplicate_frac))
        dup_idx = rng.choice(rows.index.to_numpy(), size=dup_n, replace=True)
        dup_rows = rows.loc[dup_idx].copy()
        dup_rows["experiment_row_id"] = np.arange(rows["experiment_row_id"].max() + 1, rows["experiment_row_id"].max() + 1 + len(dup_rows))
        dup_rows["experiment_assignment_delay_sec"] += rng.integers(1, 5, size=len(dup_rows))
        rows = pd.concat([rows, dup_rows], ignore_index=True)

    return rows


def simulate_subscriptions(
    config: SimulationConfig,
    users: pd.DataFrame,
    experiments: pd.DataFrame,
    rng: Generator,
    latent: LatentState,
) -> pd.DataFrame:
    """One subscription row per converting user."""
    exp_first = experiments.sort_values("experiment_row_id").drop_duplicates("user_id", keep="first")
    subscribed_map = exp_first.set_index("user_id")["subscribed"]
    subscribed_mask = users["user_id"].map(subscribed_map).fillna(0).to_numpy() == 1

    cols = [
        "subscription_id",
        "user_id",
        "experiment_id",
        "subscription_start_date",
        "subscription_end_date",
        "months_active",
        "monthly_price_paid",
        "churned",
        "churn_reason",
        "refund_flag",
        "billing_failures",
        "billing_anomaly_flag",
        "total_revenue",
        "lifetime_value",
        "plan_tier",
        "currency",
        "last_renewal_outcome",
    ]
    subs_idx = np.flatnonzero(subscribed_mask)
    if subs_idx.size == 0:
        return pd.DataFrame(columns=cols)

    u_sub = users.iloc[subs_idx].reset_index(drop=True)
    uid_s = u_sub["user_id"].to_numpy(dtype=np.int64)
    engagement = u_sub["engagement_score"].to_numpy(dtype=np.float64)

    price_map = exp_first.set_index("user_id")["assigned_price"].to_dict()
    disc_map = exp_first.set_index("user_id")["discount_pct"].to_dict()
    assigned_arr = np.array([price_map[int(u)] for u in uid_s], dtype=np.float64)
    disc_arr = np.array([disc_map[int(u)] for u in uid_s], dtype=np.float64)
    noise_u = rng.uniform(0.85, 1.05, size=len(uid_s))
    monthly_price = assigned_arr * (1.0 - (disc_arr / 100.0) * noise_u)
    monthly_price *= np.clip(1.0 + rng.normal(0.0, 0.012, size=len(uid_s)), 0.9, 1.1)

    wtp_s = latent.willingness_to_pay[subs_idx]
    intr_s = latent.intrinsic_engagement[subs_idx]
    brand_s = latent.latent_brand_affinity[subs_idx]
    sens_s = latent.price_sensitivity[subs_idx]

    n_trials = 1 + (np.round(monthly_price * 5).astype(np.int64) % 4)
    p_fail_vec = utils.clip_prob(utils.sigmoid(rng.normal(-2.9, 0.45, size=len(uid_s))), 0.01, 0.35)
    billing_failures = rng.binomial(n_trials.astype(np.int64), p_fail_vec)

    p_churn_month = build_churn_probability(
        config,
        wtp_s,
        intr_s,
        brand_s,
        monthly_price,
        engagement,
        billing_failures,
    )
    geo_draw = rng.geometric(utils.clip_prob(p_churn_month, config.churn_prob_floor, 0.985))
    months_active = np.minimum(config.max_subscription_months, geo_draw.astype(np.int32)).astype(np.int64)

    outlier_mask = rng.random(len(uid_s)) < config.outlier_ltv_frac
    if np.any(outlier_mask):
        months_active[outlier_mask] = np.minimum(
            months_active[outlier_mask] + rng.integers(12, 48, size=int(np.sum(outlier_mask))),
            config.max_subscription_months + 24,
        ).astype(np.int64)

    churned = np.ones(len(uid_s), dtype=np.int8)
    churned = np.where(months_active >= config.max_subscription_months, 0, churned)

    churn_pick = rng.choice(
        np.array(
            [
                "price_too_high",
                "did_not_use_enough",
                "competing_service",
                "billing_issue",
                "life_event",
                "seasonal_pause",
                "unknown",
            ]
        ),
        size=len(uid_s),
        p=np.array([0.18, 0.34, 0.11, 0.16, 0.09, 0.05, 0.07]),
    )
    churn_reason = churn_pick.astype(object)
    churn_reason[churned == 0] = None

    refund_prob = utils.sigmoid(
        np.where(months_active <= 2, -1.85, -3.2) + 0.9 * sens_s + rng.normal(0.0, 0.65, len(uid_s))
    )
    refund_flag = (rng.random(len(uid_s)) < refund_prob).astype(np.int8)
    refund_amt = np.where(
        refund_flag.astype(bool),
        monthly_price * rng.choice([1.0, 1.0, 2.0], size=len(uid_s), p=[0.6, 0.3, 0.1]),
        0.0,
    )

    gross = months_active.astype(np.float64) * monthly_price
    promo_credit = rng.exponential(scale=3.25, size=len(uid_s))
    promo_credit = np.where(rng.random(len(uid_s)) < 0.035, promo_credit, 0.0)
    total_rev = gross - refund_amt - promo_credit + rng.normal(0.0, 2.85, len(uid_s))
    discount_to_npv = 0.04
    ltv = total_rev / (1.0 + discount_to_npv) ** (months_active.astype(np.float64) / 12.0)
    if np.any(outlier_mask):
        boost = rng.uniform(2.8, 7.5, size=int(np.sum(outlier_mask)))
        ltv[outlier_mask] *= boost
        total_rev[outlier_mask] *= boost * rng.uniform(0.92, 1.08, size=np.sum(outlier_mask))

    billing_anomaly_flag = (rng.random(len(uid_s)) < 0.012).astype(np.int8)

    sub_start = pd.to_datetime(u_sub["signup_date"])
    sub_end = sub_start + pd.to_timedelta((months_active * 30).astype(np.int64), unit="D")

    df = pd.DataFrame(
        {
            "subscription_id": np.arange(1, len(uid_s) + 1, dtype=np.int64),
            "user_id": uid_s,
            "experiment_id": config.experiment_id,
            "subscription_start_date": sub_start.dt.strftime("%Y-%m-%d"),
            "subscription_end_date": np.where(
                churned.astype(bool),
                sub_end.dt.strftime("%Y-%m-%d"),
                pd.NA,
            ),
            "months_active": months_active,
            "monthly_price_paid": np.round(monthly_price, 2),
            "churned": churned,
            "churn_reason": churn_reason,
            "refund_flag": refund_flag,
            "billing_failures": billing_failures,
            "billing_anomaly_flag": billing_anomaly_flag,
            "total_revenue": np.round(total_rev, 2),
            "lifetime_value": np.round(ltv, 2),
            "plan_tier": utils.weighted_choice(rng, ["standard", "family", "student"], [0.78, 0.15, 0.07], len(uid_s)),
            "currency": np.where(rng.random(len(uid_s)) < 0.88, "USD", "EUR"),
            "last_renewal_outcome": utils.weighted_choice(
                rng,
                ["success", "failed_then_recovered", "failed"],
                [0.88, 0.07, 0.05],
                len(uid_s),
            ),
        }
    )
    df["months_active"] = df["months_active"].astype(np.int32)
    return df


def apply_observed_messiness(users: pd.DataFrame, subs: pd.DataFrame, rng: Generator) -> None:
    """In-place MCAR/MAR-style missingness and label noise."""
    users["age"] = users["age"].astype(float)
    users.loc[rng.random(len(users)) < 0.055, "age"] = np.nan
    users.loc[rng.random(len(users)) < 0.085, "income_segment"] = np.nan
    if not subs.empty:
        mask = (subs["churned"] == 1) & (rng.random(len(subs)) < 0.18)
        subs.loc[mask, "churn_reason"] = np.nan


def run_simulation(config: SimulationConfig) -> dict[str, pd.DataFrame]:
    """Run full pipeline; returns dict with users, pricing_experiments, subscriptions."""
    rng = _rng(config)
    utils.set_global_seed(config.random_seed)
    study_start = pd.Timestamp(config.study_start)
    study_end = pd.Timestamp(config.study_end)

    n = config.n_users
    latent = generate_latent(n, rng)
    treatment_group, assigned_price, discount_pct = assign_treatment(n, rng)

    users = simulate_users(n, rng, latent, study_start, study_end)
    experiments = simulate_pricing_experiments(
        config, users, rng, latent, treatment_group, assigned_price, discount_pct
    )
    subs = simulate_subscriptions(
        config, users, experiments, rng, latent
    )
    apply_observed_messiness(users, subs, rng)
    return {"users": users, "pricing_experiments": experiments, "subscriptions": subs}
