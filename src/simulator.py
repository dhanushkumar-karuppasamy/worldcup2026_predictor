# src/simulator.py
# ─────────────────────────────────────────────────────────────
# FIFA World Cup 2026 Monte Carlo Simulator
#
# Simulates the full 48-team tournament structure:
#   Group stage (12 groups × 4 teams × 3 matches)
#   → top 2 from each group + 8 best 3rd-place → Round of 32
#   → Round of 16 → Quarter-finals → Semi-finals → Final
#
# Run 10,000 simulations → outputs champion % per team
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from src.wc2026_config import GROUPS, ALL_TEAMS, generate_group_fixtures
from src.models import PoissonGoals, WCXGBoost, load_models

PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")


# ════════════════════════════════════════════════════════════
# TEAM FEATURES — build a feature row for any matchup
# ════════════════════════════════════════════════════════════

class TeamFeatureStore:
    """
    Holds pre-match features for every team so the simulator
    can instantly build a feature row for any matchup.

    Loaded once at startup from the processed Parquet files.
    """

    def __init__(self):
        self.elo    = self._load_elo()
        self.form   = self._load_form()
        self.feat_cols = None  # set when first match row is built

    def _load_elo(self) -> dict:
        path = PROC_DIR / "elo_ratings.parquet"
        if not path.exists():
            print("  WARNING: elo_ratings.parquet missing. Using default 1500.")
            return {}
        df = pd.read_parquet(path)
        return dict(zip(df["team"], df["elo"]))

    def _load_form(self) -> pd.DataFrame:
        path = PROC_DIR / "rolling_form.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        # Keep only the latest row per team (most recent match)
        return (df.sort_values("date")
                  .groupby("team")
                  .tail(1)
                  .set_index("team"))

    def get_elo(self, team: str) -> float:
        return self.elo.get(team, 1500.0)

    def _form_val(self, team: str, col: str) -> float:
        try:
            return float(self.form.loc[team, col])
        except (KeyError, TypeError):
            return np.nan

    def build_match_row(self, home: str, away: str,
                        stage_num: int = 1,
                        home_is_host: int = 0,
                        away_is_host: int = 0) -> pd.DataFrame:
        """
        Builds a single-row DataFrame with all features for one match.
        This is what the Poisson / XGBoost models expect as input.
        """
        eh = self.get_elo(home)
        ea = self.get_elo(away)

        row = {
            # Elo
            "elo_home_pre":    eh,
            "elo_away_pre":    ea,
            "delta_elo":       eh - ea,
            "abs_elo_diff":    abs(eh - ea),
            # Stage context
            "stage_num":       stage_num,
            "is_knockout":     int(stage_num >= 2),
            "home_is_host":    home_is_host,
            "away_is_host":    away_is_host,
            # Home rolling form
            "h_pts_avg_5":     self._form_val(home, "pts_avg_5"),
            "h_gd_avg_5":      self._form_val(home, "gd_avg_5"),
            "h_gf_avg_5":      self._form_val(home, "gf_avg_5"),
            "h_ga_avg_5":      self._form_val(home, "ga_avg_5"),
            "h_win_rate_5":    self._form_val(home, "win_rate_5"),
            "h_wpts_avg_5":    self._form_val(home, "wpts_avg_5"),
            "h_wgd_avg_5":     self._form_val(home, "wgd_avg_5"),
            "h_pts_avg_10":    self._form_val(home, "pts_avg_10"),
            "h_gd_avg_10":     self._form_val(home, "gd_avg_10"),
            "h_win_rate_10":   self._form_val(home, "win_rate_10"),
            # Away rolling form
            "a_pts_avg_5":     self._form_val(away, "pts_avg_5"),
            "a_gd_avg_5":      self._form_val(away, "gd_avg_5"),
            "a_gf_avg_5":      self._form_val(away, "gf_avg_5"),
            "a_ga_avg_5":      self._form_val(away, "ga_avg_5"),
            "a_win_rate_5":    self._form_val(away, "win_rate_5"),
            "a_wpts_avg_5":    self._form_val(away, "wpts_avg_5"),
            "a_wgd_avg_5":     self._form_val(away, "wgd_avg_5"),
            "a_pts_avg_10":    self._form_val(away, "pts_avg_10"),
            "a_gd_avg_10":     self._form_val(away, "gd_avg_10"),
            "a_win_rate_10":   self._form_val(away, "win_rate_10"),
            # Delta features
            "delta_pts_5":     self._form_val(home, "pts_avg_5")  - self._form_val(away, "pts_avg_5"),
            "delta_gd_5":      self._form_val(home, "gd_avg_5")   - self._form_val(away, "gd_avg_5"),
            "delta_wpts_5":    self._form_val(home, "wpts_avg_5") - self._form_val(away, "wpts_avg_5"),
            "delta_win_rate_5":self._form_val(home, "win_rate_5") - self._form_val(away, "win_rate_5"),
            "delta_pts_10":    self._form_val(home, "pts_avg_10") - self._form_val(away, "pts_avg_10"),
            "delta_gd_10":     self._form_val(home, "gd_avg_10")  - self._form_val(away, "gd_avg_10"),
            "competitive_edge":((self._form_val(home, "wpts_avg_5") - self._form_val(away, "wpts_avg_5"))
                                * (eh - ea) / 1000),
            # Previous WC — set to NaN for 2026 (model handles missing)
            "h_prev_pos":  np.nan, "a_prev_pos":  np.nan,
            "h_prev_pts":  np.nan, "a_prev_pts":  np.nan,
            "h_prev_gd":   np.nan, "a_prev_gd":   np.nan,
            "delta_xgd_5": np.nan,
        }
        return pd.DataFrame([row])

    def update_elo(self, home: str, away: str,
                   home_goals: int, away_goals: int, k: float = 45) -> None:
        """Updates Elo in-memory after each simulated match result."""
        eh = self.get_elo(home)
        ea = self.get_elo(away)
        ea_exp = 1 / (1 + 10 ** ((ea - eh) / 400))
        sh = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
        self.elo[home] = round(eh + k * (sh - ea_exp), 2)
        self.elo[away] = round(ea + k * ((1 - sh) - (1 - ea_exp)), 2)


# ════════════════════════════════════════════════════════════
# MATCH SIMULATOR
# ════════════════════════════════════════════════════════════

class MatchSimulator:
    """
    Simulates a single match and returns (home_goals, away_goals).
    Uses the Poisson goals model as primary engine.
    For knockout matches that end in a draw: adds extra time + shootout.
    """

    def __init__(self, poisson_model: PoissonGoals, store: TeamFeatureStore):
        self.poisson = poisson_model
        self.store   = store

    def simulate(self, home: str, away: str,
                 stage_num: int = 1,
                 home_is_host: int = 0,
                 away_is_host: int = 0,
                 knockout: bool = False) -> tuple[int, int, bool]:
        """
        Returns (home_goals, away_goals, went_to_penalties).
        In knockout mode, draws are resolved by penalty shootout (50/50).
        """
        row = self.store.build_match_row(
            home, away, stage_num, home_is_host, away_is_host
        )
        lh, la = self.poisson.predict_lambda(row)
        hg = int(np.random.poisson(max(0.1, lh[0])))
        ag = int(np.random.poisson(max(0.1, la[0])))

        penalties = False
        if knockout and hg == ag:
            # Extra time: slight random chance of a goal in ET
            et_h = np.random.poisson(0.25)
            et_a = np.random.poisson(0.25)
            hg += et_h
            ag += et_a
            if hg == ag:
                # Penalty shootout — 50/50 (can improve with team penalty data)
                penalties = True
                if np.random.random() < 0.5:
                    hg += 1
                else:
                    ag += 1

        return hg, ag, penalties

    def win_probability(self, home: str, away: str,
                        stage_num: int = 1) -> dict:
        """
        Returns match probabilities without simulation noise.
        Used for the single-match prediction output.
        """
        row = self.store.build_match_row(home, away, stage_num)
        lh, la = self.poisson.predict_lambda(row)
        ph, pd_, pa = self.poisson.match_probs(float(lh[0]), float(la[0]))
        return {
            "home":  home,
            "away":  away,
            "p_home_win": round(ph, 4),
            "p_draw":     round(pd_, 4),
            "p_away_win": round(pa, 4),
            "lambda_home": round(float(lh[0]), 2),
            "lambda_away": round(float(la[0]), 2),
        }


# ════════════════════════════════════════════════════════════
# GROUP STAGE SIMULATOR
# ════════════════════════════════════════════════════════════

def simulate_group(teams: list[str], match_sim: MatchSimulator,
                   store: TeamFeatureStore) -> pd.DataFrame:
    """
    Simulates one group (4 teams, 6 matches).
    Returns a standings DataFrame sorted by: pts → gd → gf → alphabetical.
    """
    fixtures = [
        (teams[0], teams[1]), (teams[2], teams[3]),
        (teams[0], teams[2]), (teams[1], teams[3]),
        (teams[0], teams[3]), (teams[1], teams[2]),
    ]

    stats = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0}
             for t in teams}

    for home, away in fixtures:
        hg, ag, _ = match_sim.simulate(home, away, stage_num=1, knockout=False)
        store.update_elo(home, away, hg, ag, k=45)

        # Points
        if hg > ag:
            stats[home]["pts"] += 3; stats[home]["w"] += 1; stats[away]["l"] += 1
        elif hg < ag:
            stats[away]["pts"] += 3; stats[away]["w"] += 1; stats[home]["l"] += 1
        else:
            stats[home]["pts"] += 1; stats[away]["pts"] += 1
            stats[home]["d"]   += 1; stats[away]["d"]   += 1

        # Goal stats
        for team, scored, conceded in [(home, hg, ag), (away, ag, hg)]:
            stats[team]["gf"] += scored
            stats[team]["ga"] += conceded
            stats[team]["gd"] += (scored - conceded)

    rows = [{"team": t, **v} for t, v in stats.items()]
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["pts", "gd", "gf", "team"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)
    df["position"] = df.index + 1
    return df


def simulate_all_groups(match_sim: MatchSimulator,
                        store: TeamFeatureStore) -> dict:
    """
    Simulates all 12 groups.
    Returns {group_letter: standings_df}.
    """
    results = {}
    for grp, teams in GROUPS.items():
        results[grp] = simulate_group(teams, match_sim, store)
    return results


def get_qualifiers(group_results: dict) -> dict:
    """
    Determines who qualifies from the group stage.

    2026 format (48 teams, 12 groups):
      - Top 2 from each group = 24 teams advance directly
      - Best 8 third-place teams also advance = 32 total in Round of 32

    Returns {
      "top2": {grp: [1st_team, 2nd_team]},
      "third_place": [list of 3rd-place teams sorted by pts/gd/gf],
      "r32_teams": all 32 teams entering Round of 32
    }
    """
    top2         = {}
    third_place  = []

    for grp, standings in group_results.items():
        top2[grp]     = [standings.loc[0, "team"], standings.loc[1, "team"]]
        third_row     = standings[standings["position"] == 3].iloc[0].to_dict()
        third_row["group"] = grp
        third_place.append(third_row)

    # Rank 3rd-place teams: pts → gd → gf
    third_df = pd.DataFrame(third_place).sort_values(
        ["pts", "gd", "gf"], ascending=[False, False, False]
    ).reset_index(drop=True)

    best_thirds = third_df.head(8)["team"].tolist()

    # R32 bracket: group winners vs runners-up (+ 8 best thirds)
    # Simplified seeding: pair group winner with runner-up from adjacent group
    r32_teams = []
    for grp, pair in top2.items():
        r32_teams.extend(pair)
    r32_teams.extend(best_thirds)

    return {
        "top2":        top2,
        "third_place": third_df,
        "r32_teams":   r32_teams,
        "best_thirds": best_thirds,
    }


# ════════════════════════════════════════════════════════════
# KNOCKOUT STAGE SIMULATOR
# ════════════════════════════════════════════════════════════

STAGE_NAMES = {
    32: "Round of 32",
    16: "Round of 16",
    8:  "Quarter-finals",
    4:  "Semi-finals",
    2:  "Final",
}

STAGE_NUMS = {
    "Round of 32":    2,
    "Round of 16":    3,
    "Quarter-finals": 4,
    "Semi-finals":    5,
    "Final":          6,
}


def build_r32_bracket(qualifiers: dict) -> list[tuple]:
    """
    Builds the Round of 32 matchups.

    2026 simplified bracket:
      Each group winner (1st) plays a best-third-place team or
      the runner-up from a designated group.

    For simulation purposes we use a clean pairing:
      Group A 1st vs Group B 2nd
      Group B 1st vs Group A 2nd
      Group C 1st vs Group D 2nd ...etc
      Then 8 best thirds fill in the remaining slots.

    This is approximate — FIFA hasn't published the exact bracket mapping yet.
    Update wc2026_config.py once FIFA confirms the R32 draw rules.
    """
    top2    = qualifiers["top2"]
    thirds  = qualifiers["best_thirds"]
    groups  = sorted(top2.keys())

    matchups = []
    # Pair adjacent groups: A1 vs B2, B1 vs A2, C1 vs D2, ...
    for i in range(0, len(groups) - 1, 2):
        g1, g2 = groups[i], groups[i + 1]
        matchups.append((top2[g1][0], top2[g2][1]))  # G1 winner vs G2 runner-up
        matchups.append((top2[g2][0], top2[g1][1]))  # G2 winner vs G1 runner-up

    # Pair best 8 thirds among themselves (simplified)
    for i in range(0, len(thirds) - 1, 2):
        if i + 1 < len(thirds):
            matchups.append((thirds[i], thirds[i + 1]))

    return matchups[:16]  # exactly 16 matchups for R32


def simulate_knockout_round(matchups: list[tuple],
                             match_sim: MatchSimulator,
                             store: TeamFeatureStore,
                             stage_name: str) -> list[str]:
    """
    Simulates one knockout round.
    Input:  list of (team_a, team_b) matchups
    Output: list of winners (in bracket order)
    """
    stage_num = STAGE_NUMS.get(stage_name, 2)
    winners   = []
    for home, away in matchups:
        hg, ag, _ = match_sim.simulate(
            home, away,
            stage_num=stage_num,
            knockout=True
        )
        winner = home if hg > ag else away
        winners.append(winner)
        store.update_elo(home, away, hg, ag, k=45)
    return winners


def simulate_knockout(r32_teams: list[str],
                      qualifiers: dict,
                      match_sim: MatchSimulator,
                      store: TeamFeatureStore) -> dict:
    """
    Runs the full knockout stage from R32 to Final.
    Returns a dict with the winner and all round results.
    """
    r32_matchups = build_r32_bracket(qualifiers)

    # Pad if we have fewer than 16 matchups (shouldn't happen, but safe)
    while len(r32_matchups) < 16:
        r32_matchups.append(r32_matchups[-1])

    r32_winners  = simulate_knockout_round(r32_matchups, match_sim, store, "Round of 32")
    r16_matchups = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 16, 2)]
    r16_winners  = simulate_knockout_round(r16_matchups, match_sim, store, "Round of 16")
    qf_matchups  = [(r16_winners[i], r16_winners[i + 1]) for i in range(0, 8, 2)]
    qf_winners   = simulate_knockout_round(qf_matchups,  match_sim, store, "Quarter-finals")
    sf_matchups  = [(qf_winners[i],  qf_winners[i + 1])  for i in range(0, 4, 2)]
    sf_winners   = simulate_knockout_round(sf_matchups,  match_sim, store, "Semi-finals")
    final_winner = simulate_knockout_round([tuple(sf_winners)], match_sim, store, "Final")[0]

    return {
        "r32_winners": r32_winners,
        "r16_winners": r16_winners,
        "qf_winners":  qf_winners,
        "sf_winners":  sf_winners,
        "champion":    final_winner,
    }


# ════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE
# ════════════════════════════════════════════════════════════

def run_monte_carlo(n_simulations: int = 10_000,
                    seed: int = 42) -> pd.DataFrame:
    """
    Runs the full 2026 World Cup n_simulations times.

    Each simulation:
      1. Resets Elo to pre-tournament values (cloned from store)
      2. Simulates all 12 groups
      3. Determines 32 qualifiers
      4. Simulates knockout to the final
      5. Records the champion + who reached each round

    Returns a DataFrame with probability estimates per team:
      team, champion_pct, final_pct, semifinal_pct,
      quarterfinal_pct, r16_pct, qualified_pct
    """
    np.random.seed(seed)
    print(f"\nLoading models and features...")

    # Load saved models
    lr, xgb, poisson = load_models(MODEL_DIR)
    store            = TeamFeatureStore()
    base_elo         = dict(store.elo)   # snapshot pre-tournament Elo

    # Counters
    champion_count   = defaultdict(int)
    final_count      = defaultdict(int)
    sf_count         = defaultdict(int)
    qf_count         = defaultdict(int)
    r16_count        = defaultdict(int)
    qualified_count  = defaultdict(int)

    print(f"Running {n_simulations:,} simulations...\n")

    for sim in range(n_simulations):
        if sim % 1000 == 0 and sim > 0:
            pct = 100 * sim / n_simulations
            top = sorted(champion_count.items(), key=lambda x: -x[1])[:3]
            top_str = "  |  ".join(f"{t}: {c/sim*100:.1f}%" for t,c in top)
            print(f"  Sim {sim:,}/{n_simulations:,} ({pct:.0f}%) — Top 3: {top_str}")

        # Reset Elo to base (don't let one simulation bleed into the next)
        store.elo = dict(base_elo)
        match_sim = MatchSimulator(poisson, store)

        # ── Group stage ──────────────────────────────────────
        group_results = simulate_all_groups(match_sim, store)
        qualifiers    = get_qualifiers(group_results)

        for grp, pair in qualifiers["top2"].items():
            qualified_count[pair[0]] += 1
            qualified_count[pair[1]] += 1
        for team in qualifiers["best_thirds"]:
            qualified_count[team]    += 1

        # ── Knockout stage ───────────────────────────────────
        ko = simulate_knockout(
            qualifiers["r32_teams"], qualifiers, match_sim, store
        )

        champion_count[ko["champion"]] += 1
        for team in ko["sf_winners"]:
            final_count[team]      += 1
        for team in ko["qf_winners"]:
            sf_count[team]         += 1
        for team in ko["r16_winners"]:
            qf_count[team]         += 1
        for team in ko["r32_winners"]:
            r16_count[team]        += 1

    # ── Build results DataFrame ──────────────────────────────
    rows = []
    for team in sorted(ALL_TEAMS):
        rows.append({
            "team":              team,
            "champion_pct":      round(100 * champion_count[team]  / n_simulations, 2),
            "final_pct":         round(100 * final_count[team]     / n_simulations, 2),
            "semifinal_pct":     round(100 * sf_count[team]        / n_simulations, 2),
            "quarterfinal_pct":  round(100 * qf_count[team]        / n_simulations, 2),
            "r16_pct":           round(100 * r16_count[team]       / n_simulations, 2),
            "qualified_pct":     round(100 * qualified_count[team] / n_simulations, 2),
        })

    df = pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ════════════════════════════════════════════════════════════
# SINGLE MATCH PREDICTOR (used by predict_2026.py)
# ════════════════════════════════════════════════════════════

def predict_single_match(home: str, away: str,
                         stage: str = "Group Stage") -> dict:
    """
    Returns win/draw/loss probabilities for one match.
    Quick wrapper for use during the live tournament.
    """
    _, _, poisson = load_models(MODEL_DIR)
    store     = TeamFeatureStore()
    sim       = MatchSimulator(poisson, store)
    stage_num = STAGE_NUMS.get(stage, 1)
    return sim.win_probability(home, away, stage_num)