"""
Time Charging Monte Carlo Simulation
Manufacturing Company - Direct vs Indirect Labor Analysis
3 Years of Weekly Data | 10,000 Simulations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ─── Simulation Parameters ────────────────────────────────────────────────────

RANDOM_SEED      = 42
N_SIMULATIONS    = 10_000
N_YEARS          = 3
WEEKS_PER_YEAR   = 52
TOTAL_WEEKS      = N_YEARS * WEEKS_PER_YEAR          # 156 weeks
HOURS_PER_WEEK   = 40

# Employee cohort (typical mid-size manufacturing floor)
N_EMPLOYEES      = 45

# Historical baseline assumptions derived from 3-year weekly actuals
# Direct charge mean and std per employee (as a fraction of total hours)
DIRECT_MEAN      = 0.80    # 80% direct on average
DIRECT_STD       = 0.06    # ±6% — captures production swings, shutdowns, rework

# Week-level volatility: some weeks are heavier on indirect (training, downtime)
# Modeled as a shared weekly shock on top of individual variance
WEEKLY_SHOCK_STD = 0.03    # ±3% systemic shift per week

# Seasonal effect: Q4 holiday weeks pull ~3% more indirect
SEASONAL_DIP_WEEKS   = [48, 49, 50, 51, 52]   # week-of-year indices (1-based)
SEASONAL_DIP_AMOUNT  = 0.03

np.random.seed(RANDOM_SEED)


# ─── Core Simulation ──────────────────────────────────────────────────────────

def build_week_index():
    """Create a 156-row array of week-of-year values (1–52) for seasonal tagging."""
    return np.tile(np.arange(1, WEEKS_PER_YEAR + 1), N_YEARS)[:TOTAL_WEEKS]


def simulate_single_run(week_index: np.ndarray) -> dict:
    """
    One Monte Carlo trial.

    Returns aggregated metrics across all employees and all weeks.
    """
    # Each employee draws their personal direct-fraction from the population dist.
    # Shape: (N_EMPLOYEES,)
    employee_baselines = np.random.normal(DIRECT_MEAN, DIRECT_STD, N_EMPLOYEES)
    employee_baselines = np.clip(employee_baselines, 0.40, 1.00)   # hard bounds

    # Weekly systemic shock — same direction for all employees in a given week
    # Shape: (TOTAL_WEEKS,)
    weekly_shocks = np.random.normal(0, WEEKLY_SHOCK_STD, TOTAL_WEEKS)

    # Seasonal dip for holiday weeks
    seasonal_adj = np.where(
        np.isin(week_index, SEASONAL_DIP_WEEKS), -SEASONAL_DIP_AMOUNT, 0.0
    )

    # Build the direct-fraction matrix: (N_EMPLOYEES × TOTAL_WEEKS)
    # Broadcast employee baselines (column) + weekly shocks + seasonal adjustment (row)
    direct_fractions = (
        employee_baselines[:, np.newaxis]   # (45, 1)
        + weekly_shocks[np.newaxis, :]      # (1, 156)
        + seasonal_adj[np.newaxis, :]       # (1, 156)
    )
    direct_fractions = np.clip(direct_fractions, 0.0, 1.0)

    # Convert to hours
    direct_hours   = direct_fractions * HOURS_PER_WEEK
    indirect_hours = (1 - direct_fractions) * HOURS_PER_WEEK

    total_direct   = direct_hours.sum()
    total_indirect = indirect_hours.sum()
    total_hours    = total_direct + total_indirect

    overall_direct_pct   = total_direct   / total_hours * 100
    overall_indirect_pct = total_indirect / total_hours * 100

    # Weekly aggregate for time-series insight
    weekly_direct_pct = direct_hours.mean(axis=0) / HOURS_PER_WEEK * 100

    return {
        "direct_pct":        overall_direct_pct,
        "indirect_pct":      overall_indirect_pct,
        "weekly_direct_pct": weekly_direct_pct,   # shape (156,)
        "total_direct_hrs":  total_direct,
        "total_indirect_hrs":total_indirect,
    }


def run_monte_carlo():
    """Run all 10,000 simulations and collect results."""
    week_index = build_week_index()

    direct_pcts     = np.empty(N_SIMULATIONS)
    total_dir_hrs   = np.empty(N_SIMULATIONS)
    total_indir_hrs = np.empty(N_SIMULATIONS)
    weekly_matrix   = np.empty((N_SIMULATIONS, TOTAL_WEEKS))

    print(f"Running {N_SIMULATIONS:,} Monte Carlo simulations …")
    for i in range(N_SIMULATIONS):
        result = simulate_single_run(week_index)
        direct_pcts[i]       = result["direct_pct"]
        total_dir_hrs[i]     = result["total_direct_hrs"]
        total_indir_hrs[i]   = result["total_indirect_hrs"]
        weekly_matrix[i]     = result["weekly_direct_pct"]
        if (i + 1) % 2_500 == 0:
            print(f"  … {i+1:,} / {N_SIMULATIONS:,} complete")

    print("Simulations complete.\n")
    return direct_pcts, total_dir_hrs, total_indir_hrs, weekly_matrix, week_index


# ─── Statistics ───────────────────────────────────────────────────────────────

def compute_statistics(direct_pcts: np.ndarray) -> dict:
    ci_90 = np.percentile(direct_pcts, [5,  95])
    ci_95 = np.percentile(direct_pcts, [2.5, 97.5])
    ci_99 = np.percentile(direct_pcts, [0.5, 99.5])

    normality = stats.shapiro(direct_pcts[:5_000])   # Shapiro-Wilk on first 5k

    return {
        "mean":        direct_pcts.mean(),
        "median":      np.median(direct_pcts),
        "std":         direct_pcts.std(),
        "min":         direct_pcts.min(),
        "max":         direct_pcts.max(),
        "ci_90":       ci_90,
        "ci_95":       ci_95,
        "ci_99":       ci_99,
        "skewness":    stats.skew(direct_pcts),
        "kurtosis":    stats.kurtosis(direct_pcts),
        "shapiro_p":   normality.pvalue,
        "pct_above_80": (direct_pcts >= 80).mean() * 100,
        "pct_above_75": (direct_pcts >= 75).mean() * 100,
        "pct_below_70": (direct_pcts <  70).mean() * 100,
    }


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_results(direct_pcts, weekly_matrix, week_index, stats_dict):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Monte Carlo Simulation — Direct vs Indirect Time Charging\n"
        f"Manufacturing | {N_EMPLOYEES} Direct Employees | {N_YEARS}Y Weekly Data "
        f"| {N_SIMULATIONS:,} Simulations",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    mu  = stats_dict["mean"]
    sig = stats_dict["std"]

    # ── 1. Distribution Histogram ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    n_bins = 80
    counts, bin_edges, patches = ax1.hist(
        direct_pcts, bins=n_bins, density=True,
        color="#2196F3", alpha=0.7, edgecolor="white", linewidth=0.4
    )

    # Fitted normal overlay
    x = np.linspace(direct_pcts.min(), direct_pcts.max(), 500)
    ax1.plot(x, stats.norm.pdf(x, mu, sig), "r-", lw=2.2, label="Fitted Normal")

    # CI bands
    ax1.axvline(stats_dict["ci_95"][0], color="#FF9800", ls="--", lw=1.5,
                label=f'95% CI: [{stats_dict["ci_95"][0]:.1f}%, {stats_dict["ci_95"][1]:.1f}%]')
    ax1.axvline(stats_dict["ci_95"][1], color="#FF9800", ls="--", lw=1.5)
    ax1.axvline(mu, color="#4CAF50", ls="-",  lw=2.0,
                label=f"Mean: {mu:.2f}%")
    ax1.axvline(80, color="#9C27B0", ls=":",  lw=1.8, label="Target: 80%")

    ax1.set_xlabel("Direct Time Percentage (%)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.set_title("Distribution of Simulated Direct Time %", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── 2. Summary Stats Box ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    stats_text = (
        f"{'SIMULATION SUMMARY':^28}\n"
        f"{'─'*28}\n"
        f"  Mean Direct %      {mu:>7.2f}%\n"
        f"  Median             {stats_dict['median']:>7.2f}%\n"
        f"  Std Dev            {stats_dict['std']:>7.2f}%\n"
        f"  Min / Max    {stats_dict['min']:>5.1f}% / {stats_dict['max']:>5.1f}%\n"
        f"{'─'*28}\n"
        f"  90% CI   [{stats_dict['ci_90'][0]:.2f}%, {stats_dict['ci_90'][1]:.2f}%]\n"
        f"  95% CI   [{stats_dict['ci_95'][0]:.2f}%, {stats_dict['ci_95'][1]:.2f}%]\n"
        f"  99% CI   [{stats_dict['ci_99'][0]:.2f}%, {stats_dict['ci_99'][1]:.2f}%]\n"
        f"{'─'*28}\n"
        f"  P(direct ≥ 80%)  {stats_dict['pct_above_80']:>6.1f}%\n"
        f"  P(direct ≥ 75%)  {stats_dict['pct_above_75']:>6.1f}%\n"
        f"  P(direct < 70%)  {stats_dict['pct_below_70']:>6.1f}%\n"
        f"{'─'*28}\n"
        f"  Skewness          {stats_dict['skewness']:>7.4f}\n"
        f"  Kurtosis          {stats_dict['kurtosis']:>7.4f}\n"
        f"  Shapiro-Wilk p   {stats_dict['shapiro_p']:>8.4f}\n"
    )
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=8.5, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))

    # ── 3. Weekly Time-Series (mean ± 1σ band) ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    week_nums   = np.arange(1, TOTAL_WEEKS + 1)
    weekly_mean = weekly_matrix.mean(axis=0)
    weekly_std  = weekly_matrix.std(axis=0)
    p10 = np.percentile(weekly_matrix, 10, axis=0)
    p90 = np.percentile(weekly_matrix, 90, axis=0)

    ax3.fill_between(week_nums, p10, p90, alpha=0.25, color="#2196F3",
                     label="10th–90th percentile band")
    ax3.fill_between(week_nums, weekly_mean - weekly_std,
                     weekly_mean + weekly_std, alpha=0.45, color="#2196F3",
                     label="Mean ± 1σ")
    ax3.plot(week_nums, weekly_mean, "#0D47A1", lw=1.6, label="Weekly mean")
    ax3.axhline(80, color="#9C27B0", ls=":", lw=1.5, label="80% target")
    ax3.axhline(mu, color="#4CAF50", ls="--", lw=1.3, label=f"Simulation mean {mu:.1f}%")

    # Year dividers
    for yr in range(1, N_YEARS):
        ax3.axvline(yr * WEEKS_PER_YEAR, color="gray", ls="--", lw=0.9, alpha=0.6)
        ax3.text(yr * WEEKS_PER_YEAR + 1, weekly_mean.min() - 1.5,
                 f"Year {yr+1}", fontsize=8, color="gray")

    ax3.set_xlabel("Week Number (across 3 years)", fontsize=11)
    ax3.set_ylabel("Direct Time %", fontsize=11)
    ax3.set_title("Simulated Weekly Direct Time % — Mean & Uncertainty Bands", fontsize=12)
    ax3.legend(fontsize=9, loc="lower right")
    ax3.set_xlim(1, TOTAL_WEEKS)
    ax3.grid(alpha=0.3)

    # ── 4. Q-Q Plot ────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    (osm, osr), (slope, intercept, r) = stats.probplot(direct_pcts, dist="norm")
    ax4.plot(osm, osr, "o", color="#2196F3", alpha=0.3, markersize=1.8)
    ax4.plot(osm, slope * np.array(osm) + intercept, "r-", lw=2)
    ax4.set_title("Q-Q Plot (Normality Check)", fontsize=11)
    ax4.set_xlabel("Theoretical Quantiles")
    ax4.set_ylabel("Sample Quantiles")
    ax4.text(0.05, 0.92, f"R² = {r**2:.5f}", transform=ax4.transAxes,
             fontsize=9, color="darkred")
    ax4.grid(alpha=0.3)

    # ── 5. CDF ─────────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    sorted_pcts = np.sort(direct_pcts)
    cdf = np.arange(1, N_SIMULATIONS + 1) / N_SIMULATIONS
    ax5.plot(sorted_pcts, cdf * 100, color="#2196F3", lw=2)

    for threshold, label, color in [(70, "70%", "#F44336"),
                                     (75, "75%", "#FF9800"),
                                     (80, "80%", "#9C27B0")]:
        prob = (direct_pcts >= threshold).mean() * 100
        ax5.axvline(threshold, color=color, ls="--", lw=1.3,
                    label=f"P(≥{threshold}%) = {prob:.1f}%")

    ax5.set_xlabel("Direct Time %", fontsize=11)
    ax5.set_ylabel("Cumulative Probability (%)", fontsize=11)
    ax5.set_title("CDF — Probability of Hitting Targets", fontsize=11)
    ax5.legend(fontsize=8.5)
    ax5.grid(alpha=0.3)

    # ── 6. Seasonal Pattern (avg by week-of-year) ──────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    woy_means = []
    for w in range(1, WEEKS_PER_YEAR + 1):
        mask = (week_index == w)
        woy_means.append(weekly_matrix[:, mask].mean())

    ax6.bar(range(1, WEEKS_PER_YEAR + 1), woy_means,
            color=["#F44336" if w in SEASONAL_DIP_WEEKS else "#2196F3"
                   for w in range(1, WEEKS_PER_YEAR + 1)],
            alpha=0.75, width=1.0)
    ax6.axhline(mu, color="#4CAF50", ls="--", lw=1.3, label=f"Overall mean {mu:.1f}%")
    ax6.set_xlabel("Week of Year", fontsize=10)
    ax6.set_ylabel("Avg Direct %", fontsize=10)
    ax6.set_title("Seasonal Pattern\n(red = holiday dip weeks)", fontsize=10)
    ax6.legend(fontsize=8.5)
    ax6.grid(axis="y", alpha=0.3)

    plt.savefig("time_charging_monte_carlo.png", dpi=150, bbox_inches="tight")
    print("Chart saved → time_charging_monte_carlo.png")
    plt.close()


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(stats_dict, total_dir_hrs, total_indir_hrs):
    mean_dir_hrs  = total_dir_hrs.mean()
    mean_ind_hrs  = total_indir_hrs.mean()
    annual_rate   = 75.0    # $/hr blended burdened rate (adjust as needed)
    indirect_cost = mean_ind_hrs * annual_rate / N_YEARS

    sep = "=" * 62

    print(sep)
    print("  TIME CHARGING MONTE CARLO — EXECUTIVE SUMMARY")
    print(f"  Manufacturing | {N_EMPLOYEES} Direct Employees | {N_YEARS}-Year Weekly Data")
    print(f"  {N_SIMULATIONS:,} Simulations | Normal Distribution Assumed")
    print(sep)
    print(f"\n  CENTRAL ESTIMATE")
    print(f"    Mean Direct %   :  {stats_dict['mean']:.2f}%")
    print(f"    Mean Indirect % :  {100 - stats_dict['mean']:.2f}%")
    print(f"    Std Deviation   :  {stats_dict['std']:.2f}%")

    print(f"\n  CONFIDENCE INTERVALS (Direct %)")
    print(f"    90% CI  :  [{stats_dict['ci_90'][0]:.2f}%, {stats_dict['ci_90'][1]:.2f}%]")
    print(f"    95% CI  :  [{stats_dict['ci_95'][0]:.2f}%, {stats_dict['ci_95'][1]:.2f}%]")
    print(f"    99% CI  :  [{stats_dict['ci_99'][0]:.2f}%, {stats_dict['ci_99'][1]:.2f}%]")

    print(f"\n  TARGET ACHIEVEMENT PROBABILITY")
    print(f"    P(direct ≥ 80%) :  {stats_dict['pct_above_80']:>5.1f}%")
    print(f"    P(direct ≥ 75%) :  {stats_dict['pct_above_75']:>5.1f}%")
    print(f"    P(direct < 70%) :  {stats_dict['pct_below_70']:>5.1f}%  ← risk zone")

    print(f"\n  ANNUAL HOUR & COST ESTIMATE (mean simulation)")
    print(f"    Direct Hrs/yr   :  {mean_dir_hrs/N_YEARS:>9,.0f} hrs")
    print(f"    Indirect Hrs/yr :  {mean_ind_hrs/N_YEARS:>9,.0f} hrs")
    print(f"    Indirect Cost @ ${annual_rate:.0f}/hr :  ${indirect_cost:>9,.0f} / yr")

    print(f"\n  NORMALITY CHECK")
    normal = stats_dict["shapiro_p"] > 0.05
    print(f"    Shapiro-Wilk p  :  {stats_dict['shapiro_p']:.4f}  "
          f"({'Normal ✓' if normal else 'Non-normal ✗'})")
    print(f"    Skewness        :  {stats_dict['skewness']:.4f}")
    print(f"    Kurtosis        :  {stats_dict['kurtosis']:.4f}")

    print(f"\n  KEY ASSUMPTIONS")
    print(f"    Employee direct % baseline :  μ={DIRECT_MEAN*100:.0f}%, σ={DIRECT_STD*100:.0f}%")
    print(f"    Weekly systemic shock      :  σ={WEEKLY_SHOCK_STD*100:.0f}%")
    print(f"    Seasonal holiday dip       :  -{SEASONAL_DIP_AMOUNT*100:.0f}% (weeks 48–52)")
    print(f"    Hours per week per emp.    :  {HOURS_PER_WEEK}")
    print(sep)


# ─── Sensitivity Analysis ─────────────────────────────────────────────────────

def sensitivity_analysis():
    """How does the mean direct % shift as we vary the input assumptions?"""
    print("\n  SENSITIVITY TABLE — Mean Direct % by Input Scenario")
    print(f"  {'Scenario':<28} {'Mean Direct %':>14}  {'P(≥80%)':>9}")
    print(f"  {'-'*55}")

    week_index = build_week_index()
    scenarios = [
        ("Baseline (μ=80%, σ=6%)",         0.80, 0.06),
        ("Optimistic (μ=83%, σ=4%)",        0.83, 0.04),
        ("Pessimistic (μ=77%, σ=8%)",       0.77, 0.08),
        ("High variance (μ=80%, σ=10%)",    0.80, 0.10),
        ("Stretch target (μ=85%, σ=5%)",    0.85, 0.05),
        ("Under-utilised (μ=72%, σ=7%)",    0.72, 0.07),
    ]

    for label, mu_s, sig_s in scenarios:
        sims = np.empty(2_000)
        for i in range(2_000):
            emp = np.clip(np.random.normal(mu_s, sig_s, N_EMPLOYEES), 0.4, 1.0)
            shocks = np.random.normal(0, WEEKLY_SHOCK_STD, TOTAL_WEEKS)
            season = np.where(np.isin(week_index, SEASONAL_DIP_WEEKS),
                              -SEASONAL_DIP_AMOUNT, 0.0)
            fracs = np.clip(emp[:, None] + shocks[None, :] + season[None, :], 0, 1)
            sims[i] = fracs.mean() * 100

        p_above_80 = (sims >= 80).mean() * 100
        print(f"  {label:<28} {sims.mean():>12.2f}%  {p_above_80:>8.1f}%")

    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  TIME CHARGING MONTE CARLO SIMULATION")
    print("=" * 62 + "\n")

    direct_pcts, total_dir_hrs, total_indir_hrs, weekly_matrix, week_index = \
        run_monte_carlo()

    stats_dict = compute_statistics(direct_pcts)
    print_report(stats_dict, total_dir_hrs, total_indir_hrs)
    sensitivity_analysis()
    plot_results(direct_pcts, weekly_matrix, week_index, stats_dict)

    # Also save a clean CSV of all simulation outcomes
    df = pd.DataFrame({
        "simulation":        np.arange(1, N_SIMULATIONS + 1),
        "direct_pct":        direct_pcts,
        "indirect_pct":      100 - direct_pcts,
        "total_direct_hrs":  total_dir_hrs,
        "total_indirect_hrs":total_indir_hrs,
    })
    df.to_csv("simulation_results.csv", index=False)
    print("Results saved → simulation_results.csv\n")


if __name__ == "__main__":
    main()
