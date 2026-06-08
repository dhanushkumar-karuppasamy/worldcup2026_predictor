# src/features.py
# ─────────────────────────────────────────────────────────────
# All feature engineering in one place:
#   1. Elo ratings (built from all international matches)
#   2. Rolling form (pts, GD, win rate, weighted)
#   3. xG rolling averages (from StatsBomb)
#   4. Previous WC performance features
#   5. Final feature matrix builder for modelling
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════
# 1. ELO RATINGS
# ════════════════════════════════════════════════════════════

def k_factor(tournament: str) -> float:
    """
    K-factor controls how much a match moves the Elo rating.
    Higher = more volatile. World Cup matches matter most.
    """
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return 45
    if "qualif" in t:
        return 35
    if "cup" in t or "championship" in t or "nations" in t:
        return 30
    if "friendly" in t:
        return 18
    return 24


def importance_weight(tournament: str) -> float:
    """
    Used to weight rolling form features.
    A win in a World Cup means more than a win in a friendly.
    """
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return 1.0
    if "qualif" in t:
        return 0.85
    if "cup" in t or "championship" in t or "nations" in t:
        return 0.75
    if "friendly" in t:
        return 0.35
    return 0.50


def _elo_update(ra: float, rb: float, score_a: float, k: float) -> tuple[float, float]:
    """
    Standard Elo update. score_a = 1 (win), 0.5 (draw), 0 (loss).
    Returns updated (ra, rb).
    """
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    new_ra = round(ra + k * (score_a - ea), 2)
    new_rb = round(rb + k * ((1 - score_a) - (1 - ea)), 2)
    return new_ra, new_rb


def build_elo(results: pd.DataFrame) -> pd.DataFrame:
    """
    Builds Elo ratings across ALL international matches chronologically.

    Input: results DataFrame with columns:
        date, home_team, away_team, home_score, away_score, tournament, neutral

    Returns: results with added columns:
        elo_home_pre, elo_away_pre, delta_elo
        (ratings BEFORE the match, so no data leakage)
    """
    print("Building Elo ratings...")
    df = results.copy()
    df["result"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["home_score"] < r["away_score"] else "D"),
        axis=1
    )
    df["imp"] = df["tournament"].apply(importance_weight)
    df["k"]   = df["tournament"].apply(k_factor)
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    elo_map  = {}   # team → current Elo
    elo_rows = []

    for _, r in df.iterrows():
        h, a = r["home_team"], r["away_team"]
        eh   = elo_map.get(h, 1500.0)
        ea   = elo_map.get(a, 1500.0)

        # Home advantage bump (not applied at neutral venues)
        home_bump = 0 if r.get("neutral", False) else 50
        eh_adj    = eh + home_bump

        sh = 1.0 if r["result"] == "H" else (0.5 if r["result"] == "D" else 0.0)

        elo_rows.append({
            "date":          r["date"],
            "home_team":     h,
            "away_team":     a,
            "elo_home_pre":  eh,
            "elo_away_pre":  ea,
            "delta_elo":     round(eh_adj - ea, 2),
        })
        elo_map[h], elo_map[a] = _elo_update(eh, ea, sh, r["k"])

    elo_hist = pd.DataFrame(elo_rows)
    df = pd.concat([df.reset_index(drop=True),
                    elo_hist[["elo_home_pre", "elo_away_pre", "delta_elo"]]], axis=1)
    print(f"  Elo built for {len(elo_map)} teams.")
    return df, elo_map


# ════════════════════════════════════════════════════════════
# 2. ROLLING FORM FEATURES
# ════════════════════════════════════════════════════════════

def build_rolling_form(results_with_elo: pd.DataFrame,
                       windows: tuple = (5, 10)) -> pd.DataFrame:
    """
    Converts match-level results into team-level rolling features.
    Uses a long format (one row per team per match) and shifts to avoid leakage.

    Returns a long DataFrame with columns:
        date, team, pts_avg_5, gd_avg_5, gf_avg_5, ga_avg_5,
        win_rate_5, wpts_avg_5, wgd_avg_5,
        pts_avg_10, gd_avg_10, win_rate_10, ...
    """
    print("Building rolling form features...")
    df = results_with_elo.copy()

    # Build long format: one row per team per match
    long_rows = []
    for _, r in df.iterrows():
        h_pts, a_pts = (
            (3, 0) if r["result"] == "H" else
            (0, 3) if r["result"] == "A" else
            (1, 1)
        )
        for team, opp, gf, ga, pts in [
            (r["home_team"], r["away_team"], r["home_score"], r["away_score"], h_pts),
            (r["away_team"], r["home_team"], r["away_score"], r["home_score"], a_pts),
        ]:
            long_rows.append({
                "date":       r["date"],
                "team":       team,
                "opponent":   opp,
                "gf":         gf,
                "ga":         ga,
                "pts":        pts,
                "neutral":    int(r.get("neutral", False)),
                "tournament": r["tournament"],
                "imp":        r["imp"],
            })

    long_df       = pd.DataFrame(long_rows).sort_values(["team", "date"]).reset_index(drop=True)
    long_df["gd"] = long_df["gf"] - long_df["ga"]
    long_df["win"]  = (long_df["pts"] == 3).astype(int)
    long_df["wpts"] = long_df["pts"]  * long_df["imp"]  # importance-weighted points
    long_df["wgd"]  = long_df["gd"]  * long_df["imp"]  # importance-weighted GD

    # Compute rolling averages per team, shift(1) prevents leakage
    out_frames = []
    for team, g in long_df.groupby("team"):
        g = g.sort_values("date").copy()
        for w in windows:
            for col, name in [
                ("pts",  f"pts_avg_{w}"),
                ("gd",   f"gd_avg_{w}"),
                ("gf",   f"gf_avg_{w}"),
                ("ga",   f"ga_avg_{w}"),
                ("win",  f"win_rate_{w}"),
                ("wpts", f"wpts_avg_{w}"),
                ("wgd",  f"wgd_avg_{w}"),
            ]:
                g[name] = g[col].shift(1).rolling(w, min_periods=1).mean()

        g["matches_played"] = np.arange(len(g))
        g["rest_days"]      = (g["date"] - g["date"].shift(1)).dt.days
        out_frames.append(g)

    long_feat = pd.concat(out_frames, ignore_index=True)
    print(f"  Rolling form: {len(long_feat)} team-match rows.")
    return long_feat


# ════════════════════════════════════════════════════════════
# 3. xG ROLLING FEATURES
# ════════════════════════════════════════════════════════════

def build_xg_features(xg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts match-level xG into rolling team-level xG features.

    Input: xg_df with columns: date, home_team, away_team, home_xg, away_xg
    Returns: long DataFrame with xgf_avg_5, xga_avg_5, xgd_avg_5 per team per date
    """
    if xg_df.empty:
        return pd.DataFrame()

    print("Building xG rolling features...")
    xg_long = []
    for _, r in xg_df.iterrows():
        xg_long.append({"date": r["date"], "team": r["home_team"], "xgf": r["home_xg"], "xga": r["away_xg"]})
        xg_long.append({"date": r["date"], "team": r["away_team"], "xgf": r["away_xg"], "xga": r["home_xg"]})

    xg_long_df = pd.DataFrame(xg_long).sort_values(["team", "date"]).reset_index(drop=True)
    out_frames = []
    for team, g in xg_long_df.groupby("team"):
        g = g.sort_values("date").copy()
        g["xgf_avg_5"] = g["xgf"].shift(1).rolling(5, min_periods=1).mean()
        g["xga_avg_5"] = g["xga"].shift(1).rolling(5, min_periods=1).mean()
        g["xgd_avg_5"] = g["xgf_avg_5"] - g["xga_avg_5"]
        out_frames.append(g)

    xg_feat = pd.concat(out_frames, ignore_index=True)
    print(f"  xG features: {len(xg_feat)} rows for {xg_feat.team.nunique()} teams.")
    return xg_feat


# ════════════════════════════════════════════════════════════
# 4. PREVIOUS WORLD CUP FEATURES
# ════════════════════════════════════════════════════════════

def build_prev_wc_features(standings: pd.DataFrame) -> pd.DataFrame:
    """
    For each team and each WC year, retrieves their PREVIOUS WC performance.
    This tells the model "this team were champions last time" etc.

    Returns DataFrame with: team, year, prev_pos, prev_pts, prev_gd
    """
    if standings.empty:
        return pd.DataFrame()

    print("Building previous WC features...")
    prev_records = []
    for team, g in standings.sort_values(["team", "year"]).groupby("team"):
        g = g.sort_values("year").reset_index(drop=True)
        for i, row in g.iterrows():
            prev = g[g["year"] < row["year"]].tail(1)
            p    = prev.iloc[0] if len(prev) else None
            prev_records.append({
                "team":     team,
                "year":     row["year"],
                "prev_pos": p["Position"] if p is not None else np.nan,
                "prev_pts": p["Points"]   if p is not None else np.nan,
                "prev_gd":  p["GD_clean"] if p is not None else np.nan,
            })

    df = pd.DataFrame(prev_records)
    print(f"  Prev WC features: {len(df)} rows.")
    return df


# ════════════════════════════════════════════════════════════
# 5. FINAL FEATURE MATRIX BUILDER
# ════════════════════════════════════════════════════════════

def assign_wc_stages(wc: pd.DataFrame, wcm: pd.DataFrame) -> pd.DataFrame:
    """
    Merges stage information into the WC match table.
    Handles historical stages from metadata + assigns stages for 2018/2022
    by match sequence position (since those aren't in the Kaggle file).
    """
    STAGE_ORDER = {
        "Group Stage": 1, "First round": 1,
        "Round of 16": 2, "Last 16": 2, "Second round": 2,
        "Quarter-finals": 3, "Quarterfinals": 3,
        "Semi-finals": 4, "Semifinals": 4,
        "Third place": 5, "Play-off for third place": 5,
        "Final": 6,
    }

    wcm_stage = (wcm[["Year", "home_team", "away_team", "Stage"]]
                 .rename(columns={"Year": "year"})
                 .drop_duplicates(subset=["year", "home_team", "away_team"]))

    wc = wc.merge(wcm_stage, on=["year", "home_team", "away_team"], how="left")

    def _pos_to_stage(pos: int) -> str:
        if pos < 48:   return "Group Stage"
        if pos < 56:   return "Round of 16"
        if pos < 60:   return "Quarter-finals"
        if pos < 62:   return "Semi-finals"
        if pos == 62:  return "Third place"
        return "Final"

    for yr in [2018, 2022]:
        mask    = wc["year"] == yr
        indices = wc[mask].sort_values("date").index.tolist()
        for rank, idx in enumerate(indices):
            if pd.isna(wc.loc[idx, "Stage"]) or str(wc.loc[idx, "Stage"]).strip() == "":
                wc.loc[idx, "Stage"] = _pos_to_stage(rank)

    wc["Stage"]      = wc["Stage"].fillna("Group Stage").astype(str)
    wc["stage_num"]  = wc["Stage"].map(STAGE_ORDER).fillna(1).astype(int)
    wc["is_knockout"] = (wc["stage_num"] >= 2).astype(int)
    return wc


def build_feature_matrix(results_with_elo: pd.DataFrame,
                          long_feat: pd.DataFrame,
                          xg_feat: pd.DataFrame,
                          prev_wc: pd.DataFrame,
                          wcm: pd.DataFrame,
                          wcs: pd.DataFrame) -> pd.DataFrame:
    """
    Assembles the final per-match feature matrix for all World Cup games.

    Steps:
      1. Filter to WC matches only
      2. Attach stages
      3. Merge rolling form (home and away separately, backward merge)
      4. Merge xG features
      5. Merge previous WC features
      6. Compute delta features
      7. Define target column

    Returns a DataFrame ready for model training.
    """
    print("Assembling feature matrix...")

    # ── WC matches only ──────────────────────────────────────
    wc = results_with_elo[results_with_elo["tournament"] == "FIFA World Cup"].copy()
    wc["year"]       = wc["date"].dt.year
    wc["result_num"] = wc["result"].map({"H": 1, "D": 0, "A": -1})
    wc = assign_wc_stages(wc, wcm)

    # ── Host and winner ──────────────────────────────────────
    wc = wc.merge(
        wcs[["Year", "winner_norm", "host_norm"]].rename(columns={"Year": "year"}),
        on="year", how="left"
    )
    wc["home_is_host"] = (wc["home_team"] == wc["host_norm"]).astype(int)
    wc["away_is_host"] = (wc["away_team"] == wc["host_norm"]).astype(int)

    # ── Rolling form merge (home) ────────────────────────────
    ROLL_COLS = [c for c in long_feat.columns
                 if c not in {"date", "team", "opponent", "gf", "ga", "pts",
                               "neutral", "tournament", "imp", "gd", "win", "wpts", "wgd"}]

    home_feat = (long_feat[["date", "team"] + ROLL_COLS]
                 .rename(columns={"team": "home_team"})
                 .sort_values("date"))
    home_feat.columns = (["date", "home_team"] + [f"h_{c}" for c in ROLL_COLS])

    away_feat = (long_feat[["date", "team"] + ROLL_COLS]
                 .rename(columns={"team": "away_team"})
                 .sort_values("date"))
    away_feat.columns = (["date", "away_team"] + [f"a_{c}" for c in ROLL_COLS])

    wc = wc.sort_values("date").reset_index(drop=True)
    wc = pd.merge_asof(wc, home_feat, on="date", by="home_team", direction="backward")
    wc = pd.merge_asof(wc, away_feat, on="date", by="away_team", direction="backward")

    # ── xG merge ─────────────────────────────────────────────
    if not xg_feat.empty:
        xg_h = (xg_feat[["date", "team", "xgf_avg_5", "xga_avg_5", "xgd_avg_5"]]
                .rename(columns={"team": "home_team",
                                 "xgf_avg_5": "h_xgf5", "xga_avg_5": "h_xga5", "xgd_avg_5": "h_xgd5"})
                .sort_values("date"))
        xg_a = (xg_feat[["date", "team", "xgf_avg_5", "xga_avg_5", "xgd_avg_5"]]
                .rename(columns={"team": "away_team",
                                 "xgf_avg_5": "a_xgf5", "xga_avg_5": "a_xga5", "xgd_avg_5": "a_xgd5"})
                .sort_values("date"))
        wc = pd.merge_asof(wc.sort_values("date"), xg_h, on="date", by="home_team", direction="backward")
        wc = pd.merge_asof(wc.sort_values("date"), xg_a, on="date", by="away_team", direction="backward")
        wc["delta_xgd_5"] = wc.get("h_xgd5", np.nan) - wc.get("a_xgd5", np.nan)
    else:
        wc["delta_xgd_5"] = np.nan

    # ── Previous WC features ─────────────────────────────────
    if not prev_wc.empty:
        hp = prev_wc.rename(columns={"team": "home_team", "prev_pos": "h_prev_pos",
                                      "prev_pts": "h_prev_pts", "prev_gd": "h_prev_gd"})
        ap = prev_wc.rename(columns={"team": "away_team", "prev_pos": "a_prev_pos",
                                      "prev_pts": "a_prev_pts", "prev_gd": "a_prev_gd"})
        wc = wc.merge(hp, on=["year", "home_team"], how="left")
        wc = wc.merge(ap, on=["year", "away_team"], how="left")

    # ── Delta (difference) features ──────────────────────────
    wc["delta_elo"]          = wc["elo_home_pre"] - wc["elo_away_pre"]
    wc["abs_elo_diff"]       = wc["delta_elo"].abs()
    wc["delta_pts_5"]        = wc.get("h_pts_avg_5",  np.nan) - wc.get("a_pts_avg_5",  np.nan)
    wc["delta_gd_5"]         = wc.get("h_gd_avg_5",   np.nan) - wc.get("a_gd_avg_5",   np.nan)
    wc["delta_wpts_5"]       = wc.get("h_wpts_avg_5", np.nan) - wc.get("a_wpts_avg_5", np.nan)
    wc["delta_win_rate_5"]   = wc.get("h_win_rate_5", np.nan) - wc.get("a_win_rate_5", np.nan)
    wc["delta_pts_10"]       = wc.get("h_pts_avg_10", np.nan) - wc.get("a_pts_avg_10", np.nan)
    wc["delta_gd_10"]        = wc.get("h_gd_avg_10",  np.nan) - wc.get("a_gd_avg_10",  np.nan)
    wc["competitive_edge"]   = wc["delta_wpts_5"] * wc["delta_elo"] / 1000

    wc = wc.reset_index(drop=True)
    print(f"  Feature matrix: {len(wc)} WC matches, {wc.shape[1]} columns.")
    return wc


# ── Canonical feature list ───────────────────────────────────
FEATURE_COLS = [
    # Elo
    "elo_home_pre", "elo_away_pre", "delta_elo", "abs_elo_diff",
    # Stage context
    "stage_num", "is_knockout", "home_is_host", "away_is_host",
    # Home rolling form
    "h_pts_avg_5", "h_gd_avg_5", "h_gf_avg_5", "h_ga_avg_5",
    "h_win_rate_5", "h_wpts_avg_5", "h_wgd_avg_5",
    "h_pts_avg_10", "h_gd_avg_10", "h_win_rate_10",
    # Away rolling form
    "a_pts_avg_5", "a_gd_avg_5", "a_gf_avg_5", "a_ga_avg_5",
    "a_win_rate_5", "a_wpts_avg_5", "a_wgd_avg_5",
    "a_pts_avg_10", "a_gd_avg_10", "a_win_rate_10",
    # Delta features (most predictive)
    "delta_pts_5", "delta_gd_5", "delta_wpts_5", "delta_win_rate_5",
    "delta_pts_10", "delta_gd_10", "competitive_edge",
    # Previous WC history
    "h_prev_pos", "a_prev_pos", "h_prev_pts", "a_prev_pts",
    "h_prev_gd",  "a_prev_gd",
    # xG
    "delta_xgd_5",
]


def get_features(wc: pd.DataFrame) -> list[str]:
    """Returns only the feature columns that actually exist in the dataframe."""
    available = [c for c in FEATURE_COLS if c in wc.columns]
    print(f"  Features available: {len(available)} / {len(FEATURE_COLS)}")
    return available