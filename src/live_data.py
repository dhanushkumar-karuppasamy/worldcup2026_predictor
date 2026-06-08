# src/live_data.py
# ─────────────────────────────────────────────────────────────
# Live data integration for the 2026 World Cup.
# Two sources:
#   1. WorldCupAPI  — official fixtures, scores, standings
#   2. Apify/SofaScore — player stats, live match data
#
# Usage:
#   from src.live_data import LiveDataFetcher
#   fetcher = LiveDataFetcher()
#   results = fetcher.get_completed_matches()
# ─────────────────────────────────────────────────────────────

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

PROC_DIR   = Path("data/processed")
CACHE_DIR  = Path("data/raw/live_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Add your keys to .env ────────────────────────────────────
# WORLDCUPAPI_KEY=your_key_here
# APIFY_KEY=your_apify_key_here


# ════════════════════════════════════════════════════════════
# 1. WORLDCUP API — Official 2026 tournament data
# ════════════════════════════════════════════════════════════

class WorldCupAPI:
    """
    Wraps the worldcupapi.com API.
    Available endpoints (from your Postman collection):
      /livescores          — current live match scores
      /fixtures?group=A    — group fixtures + results
      /standings?group=A   — group table
      /goalscorers         — top scorers
      /cards               — discipline stats

    Register at worldcupapi.com to get your free API key.
    Add WORLDCUPAPI_KEY to your .env file.
    """

    BASE = "https://api.worldcupapi.com"

    def __init__(self):
        self.key = os.getenv("WORLDCUPAPI_KEY", "")
        if not self.key:
            print("  WARNING: WORLDCUPAPI_KEY not set in .env")

    def _get(self, endpoint: str, params: dict = None) -> dict | list | None:
        """Generic GET with error handling and rate limiting."""
        url    = f"{self.BASE}/{endpoint}"
        params = params or {}
        params["key"] = self.key

        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP Error {r.status_code} for {endpoint}: {e}")
            return None
        except Exception as e:
            print(f"  Error fetching {endpoint}: {e}")
            return None

    def get_live_scores(self) -> pd.DataFrame:
        """Returns currently live match scores."""
        data = self._get("livescores")
        if not data:
            return pd.DataFrame()
        df = pd.json_normalize(data if isinstance(data, list) else [data])
        print(f"  Live scores: {len(df)} matches currently live")
        return df

    def get_group_fixtures(self, group: str) -> pd.DataFrame:
        """
        Returns all fixtures + results for a group.
        group: 'A' through 'L'
        """
        data = self._get("fixtures", {"group": group})
        if not data:
            return pd.DataFrame()
        rows = data if isinstance(data, list) else [data]
        df   = pd.json_normalize(rows)
        return df

    def get_all_fixtures(self) -> pd.DataFrame:
        """Fetches fixtures for all 12 groups and combines."""
        frames = []
        for grp in "ABCDEFGHIJKL":
            df = self.get_group_fixtures(grp)
            if not df.empty:
                df["group"] = grp
                frames.append(df)
            time.sleep(0.3)  # be polite to the API

        if not frames:
            return pd.DataFrame()

        all_fixtures = pd.concat(frames, ignore_index=True)
        print(f"  All fixtures: {len(all_fixtures)} matches fetched")
        return all_fixtures

    def get_standings(self, group: str, with_form: bool = False) -> pd.DataFrame:
        """Returns current group standings table."""
        params = {"group": group}
        if with_form:
            params["form"] = "1"
        data = self._get("standings", params)
        if not data:
            return pd.DataFrame()
        return pd.json_normalize(data if isinstance(data, list) else [data])

    def get_all_standings(self) -> pd.DataFrame:
        """Returns standings for all 12 groups."""
        frames = []
        for grp in "ABCDEFGHIJKL":
            df = self.get_standings(grp, with_form=True)
            if not df.empty:
                df["group"] = grp
                frames.append(df)
            time.sleep(0.3)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_goalscorers(self) -> pd.DataFrame:
        """Returns top goalscorers in the tournament."""
        data = self._get("goalscorers")
        if not data:
            return pd.DataFrame()
        return pd.json_normalize(data if isinstance(data, list) else [data])

    def get_completed_matches(self) -> pd.DataFrame:
        """
        Returns only completed matches with actual scores.
        Used to update the model's Elo and rolling form in real time.
        """
        all_fixtures = self.get_all_fixtures()
        if all_fixtures.empty:
            return pd.DataFrame()

        # Filter to completed matches — look for score columns
        score_col = next((c for c in all_fixtures.columns
                          if "home" in c.lower() and "score" in c.lower()), None)
        if score_col:
            completed = all_fixtures[all_fixtures[score_col].notna()].copy()
        else:
            completed = all_fixtures.copy()

        print(f"  Completed matches: {len(completed)}")
        return completed

    def save_snapshot(self) -> None:
        """Saves a timestamped snapshot of all live data to cache."""
        ts   = datetime.now().strftime("%Y%m%d_%H%M")
        snap = {
            "timestamp": ts,
            "fixtures":  self.get_all_fixtures().to_dict("records"),
            "standings": self.get_all_standings().to_dict("records"),
        }
        out = CACHE_DIR / f"snapshot_{ts}.json"
        with open(out, "w") as f:
            json.dump(snap, f, indent=2, default=str)
        print(f"  Snapshot saved → {out}")


# ════════════════════════════════════════════════════════════
# 2. APIFY / SOFASCORE — Player stats
# ════════════════════════════════════════════════════════════

class SofaScoreFetcher:
    """
    Uses Apify's parseforge/sofascore-live-scraper actor.
    Fetches:
      - Live football events
      - Player ratings per match
      - Team lineups

    Add APIFY_KEY to .env for cleaner code.
    """

    APIFY_BASE = "https://api.apify.com/v2"
    ACTOR_ID   = "parseforge~sofascore-live-scraper"

    def __init__(self):
        self.key = os.getenv("APIFY_KEY", "")
        if not self.key:
            print("  WARNING: APIFY_KEY not set in .env")

    def _run_actor(self, input_data: dict, max_wait_secs: int = 120) -> list:
        """
        Runs the Apify actor synchronously and returns results.
        """
        url = f"{self.APIFY_BASE}/acts/{self.ACTOR_ID}/run-sync-get-dataset-items"
        headers = {"Authorization": f"Bearer {self.key}"}
        params  = {"timeout": max_wait_secs, "memory": 256}

        try:
            r = requests.post(url, json=input_data, headers=headers,
                              params=params, timeout=max_wait_secs + 30)
            r.raise_for_status()
            return r.json() if isinstance(r.json(), list) else []
        except Exception as e:
            print(f"  Apify error: {e}")
            return []

    def get_live_football_events(self, max_items: int = 20) -> pd.DataFrame:
        """
        Fetches live football events from SofaScore.
        Returns match-level data including teams, scores, status.
        """
        print("  Fetching live football events from SofaScore...")
        items = self._run_actor({
            "fetchDetails": False,
            "maxItems":     max_items,
            "mode":         "events",
            "sport":        "football"
        })
        if not items:
            print("  No live events found (or API error).")
            return pd.DataFrame()

        df = pd.json_normalize(items)
        print(f"  Live events: {len(df)} matches")
        return df

    def get_match_lineups(self, event_id: str) -> dict:
        """
        Fetches player lineups for a specific SofaScore event ID.
        Returns {home_players: [...], away_players: [...]}
        """
        items = self._run_actor({
            "fetchDetails": True,
            "mode":         "lineups",
            "sport":        "football",
            "eventId":      event_id,
            "maxItems":     1,
        })
        return items[0] if items else {}

    def get_player_ratings_for_match(self, event_id: str) -> pd.DataFrame:
        """
        Fetches individual player ratings (SofaScore 1–10 scale) for a match.
        Useful for building squad strength features.
        """
        items = self._run_actor({
            "fetchDetails": True,
            "mode":         "playerRatings",
            "sport":        "football",
            "eventId":      event_id,
            "maxItems":     30,
        })
        if not items:
            return pd.DataFrame()
        df = pd.json_normalize(items)
        return df

    def build_squad_strength_features(self, team: str, n_matches: int = 5) -> dict:
        """
        Builds a simple squad strength feature for a team by averaging
        player ratings from their last N matches.

        Returns: {"team": str, "avg_player_rating": float, "squad_depth_score": float}

        NOTE: This is a v2 feature. For now it returns placeholder values.
        Full implementation requires matching SofaScore team IDs to your team names.
        """
        # Placeholder — implement after simulator is working
        return {
            "team":               team,
            "avg_player_rating":  7.0,   # SofaScore avg (6–8 is normal range)
            "squad_depth_score":  0.5,   # 0–1 normalised
        }


# ════════════════════════════════════════════════════════════
# 3. LIVE ELO UPDATER
# ════════════════════════════════════════════════════════════

class LiveEloUpdater:
    """
    Updates Elo ratings in real time as 2026 WC matches are played.

    Workflow:
      1. Load elo_ratings.parquet (built from all history up to 2025)
      2. After each match: update home and away Elo
      3. Save updated ratings for the next prediction run
    """

    ELO_PATH = PROC_DIR / "elo_ratings.parquet"
    WC_K     = 45   # World Cup K-factor

    def __init__(self):
        if self.ELO_PATH.exists():
            df = pd.read_parquet(self.ELO_PATH)
            self.elo_map = dict(zip(df["team"], df["elo"]))
        else:
            self.elo_map = {}
            print("  WARNING: elo_ratings.parquet not found. Run build_pipeline.py first.")

    def get_elo(self, team: str) -> float:
        return self.elo_map.get(team, 1500.0)

    def update(self, home: str, away: str,
               home_score: int, away_score: int,
               neutral: bool = True) -> None:
        """Update Elo after a completed match."""
        eh = self.get_elo(home)
        ea = self.get_elo(away)

        # Home advantage at non-neutral venues
        home_bump = 0 if neutral else 50
        eh_adj    = eh + home_bump

        ea_expected = 1 / (1 + 10 ** ((eh_adj - ea) / 400))
        sh = (1.0 if home_score > away_score
              else 0.5 if home_score == away_score
              else 0.0)

        self.elo_map[home] = round(eh + self.WC_K * (sh       - (1 - ea_expected)), 2)
        self.elo_map[away] = round(ea + self.WC_K * ((1 - sh) - ea_expected),       2)

    def update_from_df(self, matches_df: pd.DataFrame,
                       home_col: str = "home_team",
                       away_col: str = "away_team",
                       home_score_col: str = "home_score",
                       away_score_col: str = "away_score") -> None:
        """Batch update from a DataFrame of completed matches."""
        for _, row in matches_df.iterrows():
            try:
                self.update(
                    row[home_col], row[away_col],
                    int(row[home_score_col]), int(row[away_score_col]),
                    neutral=True  # All WC 2026 treated as neutral (USA/Canada/Mexico)
                )
            except Exception:
                continue
        print(f"  Elo updated for {len(matches_df)} matches.")

    def save(self) -> None:
        """Saves updated Elo ratings back to parquet."""
        df = pd.DataFrame([
            {"team": k, "elo": v} for k, v in sorted(self.elo_map.items())
        ])
        df.to_parquet(self.ELO_PATH, index=False)
        print(f"  Updated Elo saved ({len(df)} teams)")

    def top_teams(self, n: int = 20) -> pd.DataFrame:
        """Returns the top N teams by current Elo rating."""
        return (pd.DataFrame(self.elo_map.items(), columns=["team", "elo"])
                .sort_values("elo", ascending=False)
                .head(n)
                .reset_index(drop=True))


# ════════════════════════════════════════════════════════════
# 4. MASTER LIVE FETCHER (combines both sources)
# ════════════════════════════════════════════════════════════

class LiveDataFetcher:
    """
    One class to rule all live data.
    Use this in predict_2026.py and any daily update scripts.
    """
    def __init__(self):
        self.wc_api    = WorldCupAPI()
        self.sofascore = SofaScoreFetcher()
        self.elo       = LiveEloUpdater()

    def get_completed_matches(self) -> pd.DataFrame:
        return self.wc_api.get_completed_matches()

    def get_live_scores(self) -> pd.DataFrame:
        return self.wc_api.get_live_scores()

    def get_standings(self) -> pd.DataFrame:
        return self.wc_api.get_all_standings()

    def update_elo_from_results(self) -> None:
        """Fetch completed matches and update Elo ratings."""
        completed = self.get_completed_matches()
        if completed.empty:
            print("  No completed matches to update Elo with.")
            return
        self.elo.update_from_df(completed)
        self.elo.save()

    def daily_update(self) -> None:
        """
        Call this once per day during the tournament.
        Updates Elo, saves a snapshot, prints standings.
        """
        print(f"\n=== Daily update: {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
        self.update_elo_from_results()
        self.wc_api.save_snapshot()

        print("\nTop 10 teams by current Elo:")
        print(self.elo.top_teams(10).to_string(index=False))


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing live data connections...\n")

    # Test WorldCupAPI (needs key in .env)
    wc = WorldCupAPI()
    print("WorldCupAPI endpoints available:")
    print("  /livescores  /fixtures  /standings  /goalscorers  /cards")

    # Test Elo updater (works offline)
    elo = LiveEloUpdater()
    print(f"\nCurrent Elo — top 10 teams:")
    print(elo.top_teams(10).to_string(index=False))

    # Test SofaScore (needs Apify key, costs compute units)
    print("\nSofaScore fetcher ready.")
    print("Call fetcher.get_live_football_events() during tournament for live data.")