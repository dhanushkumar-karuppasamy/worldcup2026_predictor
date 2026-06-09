# scripts/predict_2026.py
# ─────────────────────────────────────────────────────────────
# Runs 10,000 Monte Carlo simulations of the 2026 World Cup
# and produces a full prediction report.
#
# Usage:
#   python scripts/predict_2026.py              (10,000 sims)
#   python scripts/predict_2026.py --sims 1000  (quick test)
# ─────────────────────────────────────────────────────────────

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.simulator import run_monte_carlo, predict_single_match
from src.wc2026_config import GROUPS, TEAM_TO_GROUP

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=10_000,
                   help="Number of Monte Carlo simulations (default: 10000)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════

def plot_champion_probabilities(results: pd.DataFrame, top_n: int = 20) -> None:
    """Horizontal bar chart of top-N champion probabilities."""
    top = results.head(top_n).copy()

    colors = []
    for team in top["team"]:
        grp = TEAM_TO_GROUP.get(team, "?")
        # Color by confederation (approximate by group)
        colors.append("#0F6E56")

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top["team"], top["champion_pct"], color="#0F6E56", alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xlabel("Champion probability (%)")
    ax.set_title("FIFA World Cup 2026 — Champion Probability\n(10,000 Monte Carlo simulations)",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, top["champion_pct"].max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "champion_probabilities.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved → outputs/champion_probabilities.png")


def plot_tournament_funnel(results: pd.DataFrame, top_n: int = 16) -> None:
    """
    Stacked bar showing qualification → R16 → QF → SF → Final → Champion
    for the top N teams by champion probability.
    """
    top = results.head(top_n).copy()
    stages = ["qualified_pct", "r16_pct", "quarterfinal_pct",
              "semifinal_pct", "final_pct", "champion_pct"]
    labels = ["Group stage out", "R16", "Quarter-final",
              "Semi-final", "Final", "Champion"]
    colors = ["#D3D1C7", "#85B7EB", "#378ADD", "#0F6E56", "#1D9E75", "#BA7517"]

    fig, ax = plt.subplots(figsize=(14, 7))
    prev = np.zeros(len(top))

    for i, (stage, label, color) in enumerate(zip(stages, labels, colors)):
        vals = top[stage].values
        ax.barh(top["team"], vals - prev if i > 0 else vals,
                left=prev if i > 0 else 0,
                color=color, label=label, alpha=0.9)
        prev = vals.copy() if i == 0 else vals

    ax.set_xlabel("Probability (%)")
    ax.set_title("Tournament progression probabilities — Top 16 teams", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tournament_funnel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved → outputs/tournament_funnel.png")


def plot_group_qualification(results: pd.DataFrame) -> None:
    """Bar chart of group-stage qualification % per group."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("Group stage qualification probability (%)", fontsize=13)
    axes_flat = axes.flatten()

    for ax, (grp, teams) in zip(axes_flat, GROUPS.items()):
        grp_df = results[results["team"].isin(teams)].sort_values("qualified_pct", ascending=False)
        bars = ax.bar(grp_df["team"], grp_df["qualified_pct"],
                      color=["#0F6E56" if i < 2 else "#B4B2A9" for i in range(len(grp_df))])
        ax.set_title(f"Group {grp}", fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8)

    for ax in axes_flat[len(GROUPS):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "group_qualification.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Chart saved → outputs/group_qualification.png")


# ════════════════════════════════════════════════════════════
# MATCH PREDICTION TABLE
# ════════════════════════════════════════════════════════════

def predict_group_stage_matches() -> pd.DataFrame:
    """
    Generates win/draw/loss probabilities for all 72 group stage matches.
    These are deterministic (not simulated) — the Poisson model's
    probability distribution, not a sampled result.
    """
    from src.wc2026_config import generate_group_fixtures
    fixtures = generate_group_fixtures()
    rows = []
    for f in fixtures:
        prob = predict_single_match(f["home"], f["away"], "Group Stage")
        rows.append({
            "group":      f["group"],
            "matchday":   f["matchday"],
            "home":       f["home"],
            "away":       f["away"],
            "p_home_win": f"{prob['p_home_win']*100:.1f}%",
            "p_draw":     f"{prob['p_draw']*100:.1f}%",
            "p_away_win": f"{prob['p_away_win']*100:.1f}%",
            "expected_home_goals": prob["lambda_home"],
            "expected_away_goals": prob["lambda_away"],
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 60)
    print("  FIFA WORLD CUP 2026 — PREDICTION SYSTEM")
    print("=" * 60)
    print(f"  Simulations : {args.sims:,}")
    print(f"  Random seed : {args.seed}")
    print("=" * 60)

    # ── Run Monte Carlo ──────────────────────────────────────
    results = run_monte_carlo(n_simulations=args.sims, seed=args.seed)

    # ── Print top 20 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CHAMPION PROBABILITY — TOP 20")
    print("=" * 60)
    top20 = results.head(20)[["rank", "team", "champion_pct", "final_pct",
                               "semifinal_pct", "qualified_pct"]].copy()
    top20.columns = ["Rank", "Team", "Champion%", "Final%", "Semi%", "Qualify%"]
    print(top20.to_string(index=False))

    # ── Save full results ─────────────────────────────────────
    results.to_csv(OUTPUT_DIR / "wc2026_predictions.csv", index=False)
    print(f"\n  Full results saved → outputs/wc2026_predictions.csv")

    # ── Charts ───────────────────────────────────────────────
    print("\n  Generating charts...")
    plot_champion_probabilities(results)
    plot_tournament_funnel(results)
    plot_group_qualification(results)

    # ── Group stage match predictions ────────────────────────
    print("\n  Generating group stage match predictions...")
    match_preds = predict_group_stage_matches()
    match_preds.to_csv(OUTPUT_DIR / "group_stage_predictions.csv", index=False)
    print("  Match predictions → outputs/group_stage_predictions.csv")

    print("\n  Sample — Group A predictions:")
    print(match_preds[match_preds["group"] == "A"].to_string(index=False))

    # ── Group qualification summary ───────────────────────────
    print("\n" + "=" * 60)
    print("  GROUP QUALIFICATION PROBABILITIES")
    print("=" * 60)
    for grp, teams in GROUPS.items():
        grp_df = results[results["team"].isin(teams)].sort_values("qualified_pct", ascending=False)
        print(f"\n  Group {grp}:")
        for _, row in grp_df.iterrows():
            bar = "█" * int(row["qualified_pct"] / 5)
            print(f"    {row['team']:<22} {row['qualified_pct']:5.1f}%  {bar}")

    print("\n" + "=" * 60)
    print("  PREDICTION COMPLETE")
    print("=" * 60)
    print(f"  Champion:    {results.iloc[0]['team']} ({results.iloc[0]['champion_pct']:.1f}%)")
    print(f"  Runner-up:   {results.iloc[1]['team']} ({results.iloc[1]['champion_pct']:.1f}%)")
    print(f"  3rd most likely: {results.iloc[2]['team']} ({results.iloc[2]['champion_pct']:.1f}%)")
    print("\n  All files saved to outputs/")


if __name__ == "__main__":
    main()