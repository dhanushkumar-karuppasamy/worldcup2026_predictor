# scripts/predict_2026_v2.py
# ─────────────────────────────────────────────────────────────
# Upgraded prediction script.
# Changes from v1:
#   - Uses ensemble model (XGBoost 40% + Poisson 60%)
#   - Models loaded ONCE globally — no more 72x spam
#   - FIFA ranking added as feature for teams with sparse history
#   - Cleaner output + all charts regenerated
#
# Usage:
#   python scripts/predict_2026_v2.py
#   python scripts/predict_2026_v2.py --sims 2000  (quick test)
# ─────────────────────────────────────────────────────────────

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from src.wc2026_config import GROUPS, TEAM_TO_GROUP, ALL_TEAMS, FIFA_RANK, generate_group_fixtures
from src.simulator import TeamFeatureStore, MatchSimulator, simulate_all_groups, get_qualifiers, simulate_knockout
from src.ensemble import EnsembleModel
from src.models import load_models, PoissonGoals

OUTPUT_DIR = Path("outputs")
MODEL_DIR  = Path("models")
PROC_DIR   = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# GLOBAL MODEL LOAD — happens ONCE
# ════════════════════════════════════════════════════════════

print("Loading models (once)...")
_LR, _XGB, _POISSON = load_models(MODEL_DIR)
_ENSEMBLE = EnsembleModel(xgb_weight=0.40, poisson_weight=0.60)
_ENSEMBLE.xgb     = _XGB
_ENSEMBLE.poisson = _POISSON
print("Models ready.\n")


# ════════════════════════════════════════════════════════════
# ENHANCED TEAM FEATURE STORE (adds FIFA ranking)
# ════════════════════════════════════════════════════════════

class EnhancedTeamStore(TeamFeatureStore):
    """
    Extends TeamFeatureStore with FIFA ranking features.
    Ranking difference helps for teams with few recent matches
    (e.g. New Zealand, Jamaica) where rolling form is sparse.
    """

    def build_match_row(self, home: str, away: str,
                        stage_num: int = 1,
                        home_is_host: int = 0,
                        away_is_host: int = 0) -> pd.DataFrame:

        row = super().build_match_row(home, away, stage_num, home_is_host, away_is_host)

        # Add FIFA ranking features
        rank_h = FIFA_RANK.get(home, 48)   # default to worst rank if unknown
        rank_a = FIFA_RANK.get(away, 48)
        row["h_fifa_rank"]    = rank_h
        row["a_fifa_rank"]    = rank_a
        # Lower rank number = better team, so negate for delta
        row["delta_fifa_rank"] = rank_a - rank_h   # positive = home team ranked better

        return row


# ════════════════════════════════════════════════════════════
# ENSEMBLE MATCH SIMULATOR
# ════════════════════════════════════════════════════════════

class EnsembleMatchSimulator(MatchSimulator):
    """
    Uses the ensemble model for win_probability() calls
    but still uses Poisson for scoreline simulation
    (Poisson is needed to sample goals, not just probabilities).
    """

    def __init__(self, ensemble: EnsembleModel, store: TeamFeatureStore):
        super().__init__(ensemble.poisson, store)
        self.ensemble = ensemble

    def win_probability(self, home: str, away: str, stage_num: int = 1) -> dict:
        """Returns ensemble-blended probabilities."""
        row = self.store.build_match_row(home, away, stage_num)
        proba = self.ensemble.predict_proba(row)   # shape (1, 3) → [away, draw, home]
        pa, pd_, ph = float(proba[0, 0]), float(proba[0, 1]), float(proba[0, 2])

        # Expected goals still from Poisson
        lh, la = self.ensemble.poisson.predict_lambda(row)
        return {
            "home":           home,
            "away":           away,
            "p_home_win":     round(ph, 4),
            "p_draw":         round(pd_, 4),
            "p_away_win":     round(pa, 4),
            "lambda_home":    round(float(lh[0]), 2),
            "lambda_away":    round(float(la[0]), 2),
        }


# ════════════════════════════════════════════════════════════
# MONTE CARLO WITH ENSEMBLE
# ════════════════════════════════════════════════════════════

def run_ensemble_monte_carlo(n_simulations: int = 10_000,
                             seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    store     = EnhancedTeamStore()
    base_elo  = dict(store.elo)

    champion_count  = defaultdict(int)
    final_count     = defaultdict(int)
    sf_count        = defaultdict(int)
    qf_count        = defaultdict(int)
    r16_count       = defaultdict(int)
    qualified_count = defaultdict(int)

    print(f"Running {n_simulations:,} ensemble simulations...")

    for sim in range(n_simulations):
        if sim % 1000 == 0 and sim > 0:
            top = sorted(champion_count.items(), key=lambda x: -x[1])[:3]
            top_str = "  |  ".join(f"{t}: {c/sim*100:.1f}%" for t, c in top)
            print(f"  {sim:,}/{n_simulations:,} — Top 3: {top_str}")

        store.elo = dict(base_elo)
        match_sim = EnsembleMatchSimulator(_ENSEMBLE, store)

        group_results = simulate_all_groups(match_sim, store)
        qualifiers    = get_qualifiers(group_results)

        for grp, pair in qualifiers["top2"].items():
            qualified_count[pair[0]] += 1
            qualified_count[pair[1]] += 1
        for team in qualifiers["best_thirds"]:
            qualified_count[team] += 1

        ko = simulate_knockout(qualifiers["r32_teams"], qualifiers, match_sim, store)

        champion_count[ko["champion"]] += 1
        for t in ko["sf_winners"]:  final_count[t] += 1
        for t in ko["qf_winners"]:  sf_count[t]    += 1
        for t in ko["r16_winners"]: qf_count[t]    += 1
        for t in ko["r32_winners"]: r16_count[t]   += 1

    rows = []
    for team in sorted(ALL_TEAMS):
        rows.append({
            "team":             team,
            "champion_pct":     round(100 * champion_count[team]  / n_simulations, 2),
            "final_pct":        round(100 * final_count[team]     / n_simulations, 2),
            "semifinal_pct":    round(100 * sf_count[team]        / n_simulations, 2),
            "quarterfinal_pct": round(100 * qf_count[team]        / n_simulations, 2),
            "r16_pct":          round(100 * r16_count[team]       / n_simulations, 2),
            "qualified_pct":    round(100 * qualified_count[team] / n_simulations, 2),
        })

    return (pd.DataFrame(rows)
            .sort_values("champion_pct", ascending=False)
            .reset_index(drop=True)
            .assign(rank=lambda d: d.index + 1))


# ════════════════════════════════════════════════════════════
# MATCH PREDICTIONS — models loaded once, reused 72 times
# ════════════════════════════════════════════════════════════

def predict_all_group_matches() -> pd.DataFrame:
    """
    Generates predictions for all 72 group stage matches.
    Uses the globally-loaded ensemble — no reload per match.
    """
    store    = EnhancedTeamStore()
    match_sim = EnsembleMatchSimulator(_ENSEMBLE, store)
    fixtures  = generate_group_fixtures()
    rows      = []

    for f in fixtures:
        prob = match_sim.win_probability(f["home"], f["away"], stage_num=1)
        rows.append({
            "group":                f["group"],
            "matchday":             f["matchday"],
            "home":                 f["home"],
            "away":                 f["away"],
            "p_home_win":           f"{prob['p_home_win']*100:.1f}%",
            "p_draw":               f"{prob['p_draw']*100:.1f}%",
            "p_away_win":           f"{prob['p_away_win']*100:.1f}%",
            "expected_home_goals":  prob["lambda_home"],
            "expected_away_goals":  prob["lambda_away"],
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════

def plot_champion_probabilities(results: pd.DataFrame) -> None:
    top = results.head(20).copy()
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = ["#0F6E56" if i < 3 else "#378ADD" if i < 8 else "#85B7EB"
              for i in range(len(top))]
    bars = ax.barh(top["team"], top["champion_pct"], color=colors, alpha=0.88)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xlabel("Champion probability (%)")
    ax.set_title("FIFA World Cup 2026 — Champion Probability\n(Ensemble model · XGBoost 40% + Poisson 60%)",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, top["champion_pct"].max() * 1.3)
    ax.grid(axis="x", alpha=0.25)

    from matplotlib.patches import Patch
    legend = [Patch(color="#0F6E56", label="Top 3"),
              Patch(color="#378ADD", label="Top 4–8"),
              Patch(color="#85B7EB", label="Rest")]
    ax.legend(handles=legend, fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "champion_probabilities_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → outputs/champion_probabilities_v2.png")


def plot_group_heatmap(results: pd.DataFrame) -> None:
    """
    Heatmap: groups on Y axis, stages on X axis, colour = probability.
    Quick way to see which groups are competitive vs decided.
    """
    stages = ["qualified_pct", "r16_pct", "quarterfinal_pct",
              "semifinal_pct", "final_pct", "champion_pct"]
    labels = ["Qualify", "R16", "QF", "SF", "Final", "Champion"]

    # Show top 2 teams per group for cleanliness
    rows = []
    for grp, teams in GROUPS.items():
        grp_df = results[results["team"].isin(teams)].sort_values("qualified_pct", ascending=False)
        for _, row in grp_df.head(2).iterrows():
            rows.append({"label": f"{grp}: {row['team']}", **{s: row[s] for s in stages}})

    heat_df = pd.DataFrame(rows).set_index("label")[stages]

    fig, ax = plt.subplots(figsize=(10, 14))
    im = ax.imshow(heat_df.values, aspect="auto", cmap="YlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(heat_df)))
    ax.set_yticklabels(heat_df.index, fontsize=9)
    for i in range(len(heat_df)):
        for j in range(len(stages)):
            val = heat_df.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="black" if val < 70 else "white")
    plt.colorbar(im, ax=ax, label="Probability (%)")
    ax.set_title("Tournament progression heatmap — top 2 per group", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tournament_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  → outputs/tournament_heatmap.png")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sims",    type=int, default=10_000)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--backtest", action="store_true",
                   help="Run ensemble backtest before main simulation")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  FIFA WORLD CUP 2026 — ENSEMBLE PREDICTION SYSTEM")
    print("=" * 60)

    # ── Optional: ensemble backtest ───────────────────────────
    if args.backtest:
        from src.ensemble import evaluate_ensemble_backtest
        from src.features import get_features
        wc = pd.read_parquet(PROC_DIR / "wc_features.parquet")
        feat_cols = get_features(wc)
        print("\nRunning ensemble backtest...")
        evaluate_ensemble_backtest(wc, feat_cols)
        print()

    # ── Monte Carlo ───────────────────────────────────────────
    results = run_ensemble_monte_carlo(args.sims, args.seed)

    # ── Print results ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CHAMPION PROBABILITY — TOP 20  (Ensemble model)")
    print("=" * 60)
    top20 = results.head(20)[["rank", "team", "champion_pct", "final_pct",
                               "semifinal_pct", "qualified_pct"]]
    top20.columns = ["Rank", "Team", "Champion%", "Final%", "Semi%", "Qualify%"]
    print(top20.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────
    results.to_csv(OUTPUT_DIR / "wc2026_predictions_v2.csv", index=False)
    print(f"\n  Saved → outputs/wc2026_predictions_v2.csv")

    # ── Match predictions (no reload spam) ────────────────────
    print("\n  Generating group stage match predictions...")
    match_preds = predict_all_group_matches()
    match_preds.to_csv(OUTPUT_DIR / "group_stage_predictions_v2.csv", index=False)
    print("  Saved → outputs/group_stage_predictions_v2.csv")

    # ── Charts ────────────────────────────────────────────────
    print("\n  Generating charts...")
    plot_champion_probabilities(results)
    plot_group_heatmap(results)

    print("\n" + "=" * 60)
    print(f"  Champion:        {results.iloc[0]['team']} ({results.iloc[0]['champion_pct']:.1f}%)")
    print(f"  Runner-up:       {results.iloc[1]['team']} ({results.iloc[1]['champion_pct']:.1f}%)")
    print(f"  3rd most likely: {results.iloc[2]['team']} ({results.iloc[2]['champion_pct']:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()