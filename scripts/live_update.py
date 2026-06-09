# scripts/live_update.py
# ─────────────────────────────────────────────────────────────
# Run this ONCE PER DAY during the 2026 World Cup.
#
# What it does:
#   1. Fetches completed match results from WorldCupAPI
#   2. Updates Elo ratings with those results
#   3. Re-runs the Monte Carlo simulator with updated Elo
#   4. Outputs refreshed predictions + a match-by-match tracker
#
# Usage:
#   python scripts/live_update.py               (full 10k sims)
#   python scripts/live_update.py --sims 2000   (quick refresh)
#   python scripts/live_update.py --manual       (enter results by hand)
# ─────────────────────────────────────────────────────────────

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROC_DIR   = Path("data/processed")
OUTPUT_DIR = Path("outputs")
MODEL_DIR  = Path("models")
LIVE_DIR   = Path("data/live")
OUTPUT_DIR.mkdir(exist_ok=True)
LIVE_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sims",   type=int,  default=10_000)
    p.add_argument("--seed",   type=int,  default=42)
    p.add_argument("--manual", action="store_true",
                   help="Enter match results by hand instead of fetching from API")
    return p.parse_args()


# ════════════════════════════════════════════════════════════
# MATCH RESULTS TRACKER
# ════════════════════════════════════════════════════════════

RESULTS_FILE = LIVE_DIR / "completed_matches.csv"


def load_completed_matches() -> pd.DataFrame:
    """Loads the running log of completed 2026 WC matches."""
    if RESULTS_FILE.exists():
        return pd.read_csv(RESULTS_FILE)
    # Create empty tracker with correct columns
    return pd.DataFrame(columns=[
        "date", "group", "stage", "home", "away",
        "home_goals", "away_goals", "result", "source"
    ])


def save_completed_matches(df: pd.DataFrame) -> None:
    df.to_csv(RESULTS_FILE, index=False)


def add_result_manual(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Interactive prompt to add match results by hand.
    Use this when the API key isn't set up yet or for testing.
    """
    print("\n=== Manual result entry ===")
    print("Type 'done' for home team when finished.\n")

    new_rows = []
    while True:
        home = input("Home team (or 'done'): ").strip()
        if home.lower() == "done":
            break
        away    = input("Away team: ").strip()
        h_goals = int(input(f"Goals — {home}: ").strip())
        a_goals = int(input(f"Goals — {away}: ").strip())
        group   = input("Group (A-L) or 'KO' for knockout: ").strip().upper()
        stage   = "Group Stage" if group in "ABCDEFGHIJKL" else input("Stage name: ").strip()

        result = "H" if h_goals > a_goals else ("A" if h_goals < a_goals else "D")
        new_rows.append({
            "date":       datetime.today().strftime("%Y-%m-%d"),
            "group":      group,
            "stage":      stage,
            "home":       home,
            "away":       away,
            "home_goals": h_goals,
            "away_goals": a_goals,
            "result":     result,
            "source":     "manual",
        })
        print(f"  ✓ Added: {home} {h_goals}–{a_goals} {away}\n")

    if new_rows:
        new_df   = pd.DataFrame(new_rows)
        combined = pd.concat([completed, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "home", "away"])
        return combined
    return completed


def add_result_from_api(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches completed matches from WorldCupAPI and adds any new ones.
    """
    try:
        from src.live_data import WorldCupAPI
        wc_api    = WorldCupAPI()
        api_df    = wc_api.get_completed_matches()

        if api_df.empty:
            print("  No matches returned from API.")
            return completed

        # Normalise column names (API field names may vary)
        col_map = {}
        for c in api_df.columns:
            cl = c.lower()
            if "home" in cl and "team" in cl:  col_map[c] = "home"
            if "away" in cl and "team" in cl:  col_map[c] = "away"
            if "home" in cl and "score" in cl: col_map[c] = "home_goals"
            if "away" in cl and "score" in cl: col_map[c] = "away_goals"
            if "date" in cl:                   col_map[c] = "date"
            if "group" in cl:                  col_map[c] = "group"
            if "stage" in cl:                  col_map[c] = "stage"
        api_df = api_df.rename(columns=col_map)

        # Keep only rows with valid scores
        for col in ["home_goals", "away_goals"]:
            if col in api_df.columns:
                api_df[col] = pd.to_numeric(api_df[col], errors="coerce")
        api_df = api_df.dropna(subset=["home_goals", "away_goals"])

        api_df["result"] = api_df.apply(
            lambda r: "H" if r["home_goals"] > r["away_goals"]
                      else ("A" if r["home_goals"] < r["away_goals"] else "D"),
            axis=1
        )
        api_df["source"] = "worldcupapi"

        for col in ["group", "stage"]:
            if col not in api_df.columns:
                api_df[col] = "Unknown"

        combined = pd.concat([completed, api_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["home", "away"])
        new_count = len(combined) - len(completed)
        print(f"  API: {new_count} new match(es) added.")
        return combined

    except Exception as e:
        print(f"  API fetch failed ({e}). Use --manual flag to enter results.")
        return completed


# ════════════════════════════════════════════════════════════
# ELO UPDATER
# ════════════════════════════════════════════════════════════

def update_elo_from_results(completed: pd.DataFrame) -> dict:
    """
    Rebuilds Elo from the base (pre-tournament) ratings,
    then applies every completed match in chronological order.
    Always rebuilds from base — so re-running is always safe.
    """
    base_path = PROC_DIR / "elo_ratings.parquet"
    if not base_path.exists():
        print("  ERROR: elo_ratings.parquet not found. Run build_pipeline.py first.")
        return {}

    df   = pd.read_parquet(base_path)
    elo  = dict(zip(df["team"], df["elo"]))
    K    = 45  # World Cup K-factor

    completed = completed.sort_values("date").reset_index(drop=True)
    updated = 0

    for _, row in completed.iterrows():
        home = str(row.get("home", "")).strip()
        away = str(row.get("away", "")).strip()
        try:
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
        except (ValueError, TypeError):
            continue

        eh  = elo.get(home, 1500.0)
        ea  = elo.get(away, 1500.0)
        exp = 1 / (1 + 10 ** ((ea - eh) / 400))
        sh  = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        elo[home] = round(eh + K * (sh - exp), 2)
        elo[away] = round(ea + K * ((1 - sh) - (1 - exp)), 2)
        updated += 1

    print(f"  Elo updated with {updated} completed match(es).")
    return elo


def save_live_elo(elo: dict) -> None:
    """Saves live Elo to a separate file — does NOT overwrite the base."""
    live_elo_path = LIVE_DIR / "live_elo_ratings.parquet"
    pd.DataFrame(elo.items(), columns=["team", "elo"]).to_parquet(live_elo_path, index=False)
    print(f"  Live Elo saved → {live_elo_path}")


# ════════════════════════════════════════════════════════════
# RE-RUN SIMULATOR WITH LIVE ELO
# ════════════════════════════════════════════════════════════

def run_live_simulation(live_elo: dict, completed: pd.DataFrame,
                        n_simulations: int, seed: int) -> pd.DataFrame:
    """
    Runs the Monte Carlo simulator with:
      - Live Elo ratings (reflecting actual results so far)
      - Already-completed matches treated as fixed (not re-simulated)
      - Only remaining matches simulated

    Returns updated champion probability DataFrame.
    """
    from src.simulator import (
        TeamFeatureStore, MatchSimulator, PoissonGoals,
        simulate_all_groups, get_qualifiers, simulate_knockout,
        GROUPS, ALL_TEAMS
    )
    from src.models import load_models
    from collections import defaultdict

    np.random.seed(seed)
    _, _, poisson = load_models(MODEL_DIR)

    # Build a store that uses LIVE elo instead of base
    store = TeamFeatureStore()
    store.elo = dict(live_elo)
    base_live_elo = dict(live_elo)

    # Determine which group matches are already done
    done_set = set()
    if not completed.empty:
        for _, r in completed.iterrows():
            key = frozenset([str(r.get("home", "")), str(r.get("away", ""))])
            done_set.add(key)

    print(f"  Completed matches locked in: {len(done_set)}")
    print(f"  Running {n_simulations:,} forward simulations...")

    champion_count  = defaultdict(int)
    final_count     = defaultdict(int)
    sf_count        = defaultdict(int)
    qf_count        = defaultdict(int)
    r16_count       = defaultdict(int)
    qualified_count = defaultdict(int)

    for sim in range(n_simulations):
        store.elo = dict(base_live_elo)
        match_sim = MatchSimulator(poisson, store)

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
# MATCH ACCURACY TRACKER
# ════════════════════════════════════════════════════════════

def evaluate_predictions_so_far(completed: pd.DataFrame) -> None:
    """
    Compares model predictions vs actual results for completed matches.
    Prints running accuracy as the tournament progresses.
    """
    if completed.empty or len(completed) < 3:
        print("  Not enough completed matches yet to evaluate accuracy.")
        return

    from src.simulator import TeamFeatureStore, MatchSimulator, PoissonGoals
    from src.models import load_models

    _, _, poisson = load_models(MODEL_DIR)
    store         = TeamFeatureStore()  # uses base Elo intentionally (pre-match)
    sim           = MatchSimulator(poisson, store)

    correct = 0
    rows    = []
    for _, row in completed.iterrows():
        home = str(row.get("home", ""))
        away = str(row.get("away", ""))
        actual = str(row.get("result", ""))
        if not home or not away or not actual:
            continue

        prob = sim.win_probability(home, away, stage_num=1)
        predicted = (
            "H" if prob["p_home_win"] > prob["p_draw"] and prob["p_home_win"] > prob["p_away_win"]
            else "A" if prob["p_away_win"] > prob["p_draw"] and prob["p_away_win"] > prob["p_home_win"]
            else "D"
        )
        hit = (predicted == actual)
        if hit:
            correct += 1

        rows.append({
            "match":     f"{home} vs {away}",
            "actual":    actual,
            "predicted": predicted,
            "correct":   "✓" if hit else "✗",
            "p_home":    f"{prob['p_home_win']*100:.0f}%",
            "p_draw":    f"{prob['p_draw']*100:.0f}%",
            "p_away":    f"{prob['p_away_win']*100:.0f}%",
        })

    total = len(rows)
    acc   = correct / total if total > 0 else 0

    print(f"\n  === Live accuracy: {correct}/{total} correct ({acc*100:.1f}%) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # Save to CSV
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "live_accuracy.csv", index=False)


# ════════════════════════════════════════════════════════════
# MOVEMENT CHART — how predictions changed
# ════════════════════════════════════════════════════════════

def plot_prediction_movement(current: pd.DataFrame) -> None:
    """
    Compares current predictions to the pre-tournament baseline.
    Shows which teams moved up or down after actual results.
    """
    baseline_path = OUTPUT_DIR / "wc2026_predictions.csv"
    if not baseline_path.exists():
        print("  Baseline predictions not found — skipping movement chart.")
        return

    baseline = pd.read_csv(baseline_path)[["team", "champion_pct"]].rename(
        columns={"champion_pct": "baseline_pct"}
    )
    merged = current[["team", "champion_pct"]].merge(baseline, on="team", how="left")
    merged["delta"] = merged["champion_pct"] - merged["baseline_pct"]
    merged = merged.sort_values("delta", ascending=False).head(20)

    colors = ["#0F6E56" if d >= 0 else "#C0392B" for d in merged["delta"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(merged["team"], merged["delta"], color=colors, alpha=0.85)
    ax.bar_label(bars, fmt=lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%",
                 padding=4, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in champion probability vs pre-tournament")
    ax.set_title("Prediction movement after live results", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_movement.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Chart saved → outputs/prediction_movement.png")


# ════════════════════════════════════════════════════════════
# DAILY SUMMARY REPORT
# ════════════════════════════════════════════════════════════

def print_daily_summary(completed: pd.DataFrame, results: pd.DataFrame) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"  DAILY PREDICTION UPDATE — {ts}")
    print(f"  Matches completed so far: {len(completed)}")
    print(f"{'='*60}")

    print("\n  TOP 10 CHAMPION PROBABILITIES (updated):")
    top10 = results.head(10)[["rank", "team", "champion_pct", "final_pct", "semifinal_pct"]]
    top10.columns = ["#", "Team", "Champion%", "Final%", "Semi%"]
    print(top10.to_string(index=False))

    print("\n  GROUP QUALIFICATION (updated):")
    from src.wc2026_config import GROUPS
    for grp, teams in GROUPS.items():
        grp_df = results[results["team"].isin(teams)].sort_values(
            "qualified_pct", ascending=False
        )
        print(f"\n  Group {grp}:")
        for _, row in grp_df.iterrows():
            bar = "█" * int(row["qualified_pct"] / 5)
            print(f"    {row['team']:<22} {row['qualified_pct']:5.1f}%  {bar}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 60)
    print("  WC2026 LIVE UPDATE SYSTEM")
    print("=" * 60)

    # ── Step 1: Load completed matches ───────────────────────
    completed = load_completed_matches()
    print(f"  Matches on record: {len(completed)}")

    # ── Step 2: Add new results ──────────────────────────────
    if args.manual:
        completed = add_result_manual(completed)
    else:
        completed = add_result_from_api(completed)

    save_completed_matches(completed)

    # ── Step 3: Update Elo ───────────────────────────────────
    print("\n  Updating Elo ratings from results...")
    live_elo = update_elo_from_results(completed)

    if live_elo:
        save_live_elo(live_elo)
        # Print top 10 movers
        base_df = pd.read_parquet(PROC_DIR / "elo_ratings.parquet")
        base_elo = dict(zip(base_df["team"], base_df["elo"]))
        movers = []
        for team, new_elo in live_elo.items():
            old_elo = base_elo.get(team, 1500.0)
            movers.append({"team": team, "old": old_elo, "new": new_elo,
                           "delta": round(new_elo - old_elo, 1)})
        movers_df = pd.DataFrame(movers).sort_values("delta", ascending=False)
        print("\n  Top Elo movers (tournament so far):")
        print(movers_df.head(8)[["team", "old", "new", "delta"]].to_string(index=False))

    # ── Step 4: Evaluate live accuracy ───────────────────────
    print("\n  Evaluating prediction accuracy on completed matches...")
    evaluate_predictions_so_far(completed)

    # ── Step 5: Re-simulate with updated Elo ─────────────────
    print(f"\n  Re-running Monte Carlo ({args.sims:,} simulations)...")
    if live_elo:
        updated_results = run_live_simulation(
            live_elo, completed, args.sims, args.seed
        )
    else:
        # Fallback to fresh simulation if Elo update failed
        from src.simulator import run_monte_carlo
        updated_results = run_monte_carlo(args.sims, args.seed)

    # ── Step 6: Save + chart ─────────────────────────────────
    ts_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"wc2026_live_{ts_str}.csv"
    updated_results.to_csv(out_path, index=False)
    print(f"  Updated predictions saved → {out_path}")

    plot_prediction_movement(updated_results)

    # ── Step 7: Daily summary ────────────────────────────────
    print_daily_summary(completed, updated_results)

    print(f"\n{'='*60}")
    print("  Run again tomorrow after more matches to update.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()