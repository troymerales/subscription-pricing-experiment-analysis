"""CLI and programmatic export of scalable synthetic CSV outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import utils
from .simulation import SimulationConfig, run_simulation


def summarize(dfs: dict[str, Any], config: SimulationConfig) -> dict[str, Any]:
    users = dfs["users"]
    exp = dfs["pricing_experiments"].drop_duplicates("user_id", keep="first")
    subs = dfs["subscriptions"]
    conv_rate = float(exp["subscribed"].mean())
    summary: dict[str, Any] = {
        "random_seed": config.random_seed,
        "n_users": int(len(users)),
        "n_experiment_rows": int(len(dfs["pricing_experiments"])),
        "n_unique_users_in_experiment": int(exp.shape[0]),
        "overall_conversion_rate": round(conv_rate, 5),
        "n_subscriptions": int(0 if subs.empty else len(subs)),
        "conversion_by_treatment": exp.groupby("treatment_group")["subscribed"].mean().round(4).to_dict(),
    }
    if not subs.empty:
        summary.update(
            {
                "subscriber_churn_rate": float((subs["churned"] == 1).mean()),
                "median_ltv_usd_approx": float(subs.loc[subs["currency"] == "USD", "lifetime_value"].median())
                if (subs["currency"] == "USD").any()
                else float(subs["lifetime_value"].median()),
                "median_months_active": float(subs["months_active"].median()),
                "refund_share": float(subs["refund_flag"].mean()),
            }
        )
    return summary


def export_csvs(dfs: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    utils.ensure_dir(str(out))
    dfs["users"].to_csv(out / "users.csv", index=False)
    dfs["pricing_experiments"].to_csv(out / "pricing_experiments.csv", index=False)
    if dfs["subscriptions"].empty:
        dfs["subscriptions"].to_csv(out / "subscriptions.csv", index=False)
    else:
        dfs["subscriptions"].to_csv(out / "subscriptions.csv", index=False)


def generate(
    n_users: int,
    seed: int,
    output_dir: str | Path,
    write_summary: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run simulation and write CSVs + optional summary JSON."""
    cfg = SimulationConfig(n_users=n_users, random_seed=seed, **kwargs)
    t0 = time.perf_counter()
    dfs = run_simulation(cfg)
    export_csvs(dfs, output_dir)
    summary = summarize(dfs, cfg)
    summary["runtime_seconds"] = round(time.perf_counter() - t0, 3)
    if write_summary:
        utils.write_json(str(Path(output_dir) / "generation_summary.json"), summary)
    return {"tables": dfs, "summary": summary, "config": cfg}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic subscription pricing experiment data generator")
    parser.add_argument("--n-rows", type=int, default=10_000, help="Number of users (rows in users table)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "output"),
    )
    parser.add_argument("--no-duplicates", action="store_true", help="Disable duplicate experiment log rows")
    parser.add_argument("--max-months", type=int, default=60)
    args = parser.parse_args(argv)

    generate(
        n_users=args.n_rows,
        seed=args.seed,
        output_dir=args.output_dir,
        include_duplicate_experiment_rows=not args.no_duplicates,
        max_subscription_months=args.max_months,
    )
    print(json.dumps({"status": "ok", "output_dir": args.output_dir, "n_rows": args.n_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
