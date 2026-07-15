"""
Offline A/B evaluation framework (Day 4, part 2).

Compares the CF baseline's recommendation policy against the uplift model's
recommendation policy WITHOUT live traffic, using Inverse Propensity
Scoring (IPS). This is the standard technique for evaluating a new
recommendation policy against logged data collected under a DIFFERENT
(confounded) policy.

How IPS works here:
  - We have logged data: for each user, which treatment the OLD policy
    picked (`treatment`), the propensity of that choice (`propensity`,
    from the logged data -- this is the one place `propensity` is a
    legitimate input, unlike as a model feature), and the realized outcome.
  - For a NEW policy pi (e.g. "recommend whatever the uplift model says"),
    the IPS estimator reweights logged outcomes by 1{pi(x) == treatment} / propensity
    to get an unbiased estimate of "what would the average outcome have
    been if pi had been running all along."
  - We clip propensities to avoid the classic IPS variance blowup from
    near-zero propensities (already bounded [0.05, 0.95] at generation time,
    but we clip again defensively here since a real logged policy might not
    be so well-behaved).

Output: data/processed/offline_ab_report.csv + printed summary.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROPENSITY_CLIP = (0.02, 0.98)


def ips_policy_value(df: pd.DataFrame, policy_col: str) -> dict:
    """Compute the IPS estimate of a policy's average outcome.

    df must have: treatment, propensity, outcome_30d_return_pct, policy_col
    """
    logged_t = df["treatment"].values
    propensity = df["propensity"].clip(*PROPENSITY_CLIP).values
    outcome = df["outcome_30d_return_pct"].values
    policy_t = df[policy_col].values

    matches = (policy_t == logged_t).astype(float)
    # P(logged_t | X) under the policy's chosen action:
    # if policy picks treatment=1, use propensity; if picks 0, use (1-propensity)
    prob_of_policy_choice_under_logging = np.where(policy_t == 1, propensity, 1 - propensity)

    weights = matches / prob_of_policy_choice_under_logging
    ips_estimate = np.mean(weights * outcome)

    # Effective sample size (diagnostic: low ESS = high-variance / unreliable estimate)
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0

    return {
        "ips_estimate": ips_estimate,
        "match_rate": matches.mean(),
        "effective_sample_size": ess,
        "n": len(df),
    }


def logged_policy_value(df: pd.DataFrame) -> float:
    """What the ORIGINAL logged policy actually achieved (no reweighting needed --
    this is just the observed average outcome under the policy that was running)."""
    return df["outcome_30d_return_pct"].mean()


def bootstrap_ips_diff(df_cf: pd.DataFrame, df_uplift: pd.DataFrame, n_boot: int = 2000, seed: int = 42):
    """Bootstrap the difference in IPS estimates (uplift - cf) to get a
    confidence interval, since raw point estimates can look different
    just from noise at n=2000. Resamples user_id rows jointly (same
    resample index applied to both dataframes, since they're paired by user)."""
    rng = np.random.default_rng(seed)
    n = len(df_cf)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cf_boot = df_cf.iloc[idx]
        up_boot = df_uplift.iloc[idx]
        cf_val = ips_policy_value(cf_boot, "cf_recommended_treatment")["ips_estimate"]
        up_val = ips_policy_value(up_boot, "uplift_recommended_treatment")["ips_estimate"]
        diffs.append(up_val - cf_val)
    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [5, 95])
    return {"mean_diff": diffs.mean(), "ci_low_90": ci_low, "ci_high_90": ci_high}


def main():
    cf = pd.read_csv(PROCESSED_DIR / "cf_baseline_predictions.csv")
    uplift = pd.read_csv(PROCESSED_DIR / "uplift_predictions.csv")
    rec_log = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "raw" / "recommendations_log.csv")

    # Merge propensity (only available in the raw log) into both prediction sets
    cf = cf.merge(rec_log[["user_id", "propensity"]], on="user_id")
    uplift = uplift.merge(rec_log[["user_id", "propensity"]], on="user_id")

    logged_value = logged_policy_value(cf)
    cf_result = ips_policy_value(cf, "cf_recommended_treatment")
    uplift_result = ips_policy_value(uplift, "uplift_recommended_treatment")

    print("=" * 60)
    print("OFFLINE A/B EVALUATION REPORT (IPS)")
    print("=" * 60)
    print(f"\nLogged (existing) policy observed average outcome: {logged_value:.4f}")
    print(f"\nCF baseline policy:")
    print(f"  IPS-estimated value:      {cf_result['ips_estimate']:.4f}")
    print(f"  Match rate w/ logged:     {cf_result['match_rate']:.2%}")
    print(f"  Effective sample size:    {cf_result['effective_sample_size']:.1f} / {cf_result['n']}")
    print(f"\nUplift (causal) model policy:")
    print(f"  IPS-estimated value:      {uplift_result['ips_estimate']:.4f}")
    print(f"  Match rate w/ logged:     {uplift_result['match_rate']:.2%}")
    print(f"  Effective sample size:    {uplift_result['effective_sample_size']:.1f} / {uplift_result['n']}")

    lift_vs_logged_cf = cf_result["ips_estimate"] - logged_value
    lift_vs_logged_uplift = uplift_result["ips_estimate"] - logged_value
    print(f"\nEstimated lift vs. logged policy:")
    print(f"  CF baseline:    {lift_vs_logged_cf:+.4f} pp")
    print(f"  Uplift model:   {lift_vs_logged_uplift:+.4f} pp")

    if uplift_result["ips_estimate"] > cf_result["ips_estimate"]:
        print(f"\n=> Uplift model wins the offline A/B by {uplift_result['ips_estimate'] - cf_result['ips_estimate']:+.4f} pp (IPS-estimated).")
    else:
        print(f"\n=> CF baseline wins the offline A/B by {cf_result['ips_estimate'] - uplift_result['ips_estimate']:+.4f} pp (IPS-estimated) -- checking significance below.")

    print("\nBootstrapping (2000 resamples) to check if that gap is distinguishable from noise...")
    boot = bootstrap_ips_diff(cf, uplift)
    print(f"  Mean (uplift - cf) IPS diff: {boot['mean_diff']:+.4f}")
    print(f"  90% CI: [{boot['ci_low_90']:+.4f}, {boot['ci_high_90']:+.4f}]")
    if boot["ci_low_90"] <= 0 <= boot["ci_high_90"]:
        print("  -> CI spans zero: the two policies are NOT statistically distinguishable")
        print("     at this sample size. Report this as 'no significant difference detected'")
        print("     rather than declaring a winner -- that's the honest, defensible conclusion.")
    else:
        print("  -> CI excludes zero: the difference is statistically significant at 90% confidence.")

    if min(cf_result["effective_sample_size"], uplift_result["effective_sample_size"]) < 0.3 * len(cf):
        print("\nCAUTION: effective sample size is low relative to n -- IPS variance is high here,")
        print("treat the point estimate with a wide error bar rather than as precise.")

    report = pd.DataFrame([
        {"policy": "logged_existing", "ips_estimate": logged_value, "match_rate": 1.0,
         "effective_sample_size": len(cf), "n": len(cf)},
        {"policy": "cf_baseline", **cf_result},
        {"policy": "uplift_causal_model", **uplift_result},
    ])
    report["ips_diff_uplift_minus_cf_mean"] = boot["mean_diff"]
    report["ips_diff_ci_low_90"] = boot["ci_low_90"]
    report["ips_diff_ci_high_90"] = boot["ci_high_90"]
    report.to_csv(PROCESSED_DIR / "offline_ab_report.csv", index=False)
    print(f"\nWrote {PROCESSED_DIR / 'offline_ab_report.csv'}")


if __name__ == "__main__":
    main()
