# src/data_loader.py
# ─────────────────────────────────────────────────────────────
# Downloads all datasets once, caches to data/raw/.
# Call load_all_data() from any script — it handles the rest.
# ─────────────────────────────────────────────────────────────

import os
import json
import glob
import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Team name resolver ───────────────────────────────────────
# Standardises historical and regional name variants to a single form.
# Add new aliases here whenever a merge produces unexpected NaN values.
KNOWN_MAP = {
    "West Germany": "Germany", "Germany FR": "Germany",
    "Soviet Union": "Russia", "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia", "Yugoslavia": "Serbia",
    "Czechoslovakia": "Czech Republic",
    "Korea Republic": "South Korea", "Korea DPR": "North Korea",
    "IR Iran": "Iran", "China PR": "China",
    "USA": "United States", "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire", "C?te d'Ivoire": "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "North Macedonia": "Macedonia", "Republic of Ireland": "Ireland",
    "Dutch East Indies": "Indonesia", "Zaire": "DR Congo",
    "Russian Federation": "Russia", "East Germany": "Germany",
    "Türkiye": "Turkey", "Turkiye": "Turkey",
    "Cape Verde Islands": "Cape Verde",
    "São Tomé and Príncipe": "Sao Tome and Principe",
}
_alias_lower = {re.sub(r"[^\w\s]", "", k.lower()).strip(): v
                for k, v in KNOWN_MAP.items()}


def resolve_team(name: str) -> str:
    """Return canonical team name. Falls back to the original if unknown."""
    name = str(name).strip()
    if name in KNOWN_MAP:
        return KNOWN_MAP[name]
    key = re.sub(r"[^\w\s]", "", name.lower()).strip()
    return _alias_lower.get(key, name)


# ── CSV reader that handles encoding issues ──────────────────
def read_csv_safe(path: str | Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {path} with any known encoding.")


# ── Dataset downloaders ──────────────────────────────────────
def _download_kaggle(slug: str, dest_name: str) -> Path:
    """
    Download a Kaggle dataset and return the local directory path.
    Requires KAGGLE_USERNAME and KAGGLE_KEY in .env
    """
    import kagglehub
    dest = RAW_DIR / dest_name
    if dest.exists() and any(dest.iterdir()):
        print(f"  [cache] {dest_name} already downloaded.")
        return dest

    print(f"  [download] {slug} ...")
    path = kagglehub.dataset_download(slug)
    # kagglehub downloads to its own cache; we copy a reference
    dest.mkdir(parents=True, exist_ok=True)
    # Write a path reference file so we can find it later
    (dest / "_kagglehub_path.txt").write_text(str(path))
    print(f"  [done] cached at {path}")
    return Path(path)


def _get_kaggle_path(dest_name: str, slug: str) -> Path:
    """Return the actual data directory, downloading if needed."""
    ref_file = RAW_DIR / dest_name / "_kagglehub_path.txt"
    if ref_file.exists():
        return Path(ref_file.read_text().strip())
    return _download_kaggle(slug, dest_name)


# ── Public loader ────────────────────────────────────────────
def load_international_results() -> pd.DataFrame:
    """
    Loads the martj42 international results dataset.
    Returns a clean DataFrame with resolved team names.
    """
    path = _get_kaggle_path(
        "intl_results",
        "martj42/international-football-results-from-1872-to-2017"
    )
    df = read_csv_safe(Path(path) / "results.csv")
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"]  = df["home_team"].apply(resolve_team)
    df["away_team"]  = df["away_team"].apply(resolve_team)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"]    = df["neutral"].fillna(False).astype(bool)
    df = df.dropna(subset=["date", "home_score", "away_score"])
    print(f"  [intl_results] {len(df)} matches loaded, up to {df.date.max().date()}")
    return df


def load_wc_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads WC match-level and tournament-level metadata.
    Returns (wc_matches_df, wc_tournaments_df).
    """
    path = _get_kaggle_path(
        "wc_abecklas",
        "abecklas/fifa-world-cup"
    )
    wcm = read_csv_safe(Path(path) / "WorldCupMatches.csv")
    wcs = read_csv_safe(Path(path) / "WorldCups.csv")

    # Patch in 2018 and 2022 which may be missing
    patch = pd.DataFrame({
        "Year": [2018, 2022],
        "Country": ["Russia", "Qatar"],
        "Winner": ["France", "Argentina"],
    })
    wcs = pd.concat([wcs, patch], ignore_index=True)
    wcs["Year"]        = pd.to_numeric(wcs["Year"], errors="coerce").astype("Int64")
    wcs["winner_norm"] = wcs["Winner"].apply(resolve_team)
    wcs["host_norm"]   = wcs["Country"].apply(resolve_team)

    wcm = wcm.dropna(subset=["Home Team Name", "Away Team Name"]).copy()
    wcm["Year"]      = pd.to_numeric(wcm["Year"], errors="coerce").astype("Int64")
    wcm["home_team"] = wcm["Home Team Name"].apply(resolve_team)
    wcm["away_team"] = wcm["Away Team Name"].apply(resolve_team)

    print(f"  [wc_metadata] {len(wcm)} WC match rows, {len(wcs)} tournament rows")
    return wcm, wcs


def load_wc_standings() -> pd.DataFrame:
    """
    Loads per-year WC group stage standings (iamsouravbanerjee dataset).
    """
    path = _get_kaggle_path(
        "wc_standings",
        "iamsouravbanerjee/fifa-football-world-cup-dataset"
    )
    frames = []
    for fpath in sorted(glob.glob(str(Path(path) / "FIFA - [0-9]*.csv"))):
        yr_match = re.search(r"(\d{4})", os.path.basename(fpath))
        if not yr_match:
            continue
        yr = int(yr_match.group(1))
        df = read_csv_safe(fpath)
        df["year"] = yr
        frames.append(df)

    if not frames:
        print("  [wc_standings] WARNING: no standing files found.")
        return pd.DataFrame()

    stand = pd.concat(frames, ignore_index=True)
    stand["team"] = stand["Team"].apply(resolve_team)

    def parse_gd(x):
        try:
            return int(str(x).replace("−", "-").replace("\u2212", "-").replace("\u2013", "-"))
        except Exception:
            return np.nan

    stand["GD_clean"] = stand["Goal Difference"].apply(parse_gd)
    for c in ["Position", "Games Played", "Points"]:
        stand[c] = pd.to_numeric(stand[c], errors="coerce")

    print(f"  [wc_standings] {len(stand)} team-year rows")
    return stand


def load_statsbomb_xg() -> pd.DataFrame:
    """
    Loads StatsBomb open data xG from the Kaggle mirror.
    Returns a DataFrame with columns:
        date, home_team, away_team, home_xg, away_xg
    """
    path = _get_kaggle_path(
        "statsbomb",
        "saurabhshahane/statsbomb-football-data"
    )
    # Find the data directory that contains competitions.json
    sb_data_dir = None
    for root, dirs, files in os.walk(str(path)):
        if "competitions.json" in files:
            sb_data_dir = root
            break

    if not sb_data_dir:
        print("  [statsbomb] WARNING: competitions.json not found.")
        return pd.DataFrame()

    comp_file = Path(sb_data_dir) / "competitions.json"
    with open(comp_file, encoding="utf-8") as f:
        comps = json.load(f)

    wc_comps = [
        c for c in comps
        if "World Cup" in c.get("competition_name", "")
        and c.get("competition_gender") == "male"
    ]

    records = []
    for comp in wc_comps:
        cid, sid = comp["competition_id"], comp["season_id"]
        matches_file = Path(sb_data_dir) / "matches" / str(cid) / f"{sid}.json"
        if not matches_file.exists():
            continue

        with open(matches_file, encoding="utf-8") as f:
            matches = json.load(f)

        for m in matches:
            mid    = m["match_id"]
            hteam  = resolve_team(m["home_team"]["home_team_name"])
            ateam  = resolve_team(m["away_team"]["away_team_name"])
            mdate  = m["match_date"]

            events_file = Path(sb_data_dir) / "events" / f"{mid}.json"
            if not events_file.exists():
                continue

            with open(events_file, encoding="utf-8") as f:
                events = json.load(f)

            h_xg, a_xg = 0.0, 0.0
            for ev in events:
                if ev.get("type", {}).get("name") == "Shot" and "shot" in ev:
                    xg    = ev["shot"].get("statsbomb_xg", 0.0)
                    tname = resolve_team(ev.get("team", {}).get("name", ""))
                    if tname == hteam:
                        h_xg += xg
                    elif tname == ateam:
                        a_xg += xg

            records.append({
                "date":      pd.to_datetime(mdate),
                "home_team": hteam,
                "away_team": ateam,
                "home_xg":   h_xg,
                "away_xg":   a_xg,
            })

    xg_df = pd.DataFrame(records)
    print(f"  [statsbomb] {len(xg_df)} xG match records loaded")
    return xg_df


def load_all_data() -> dict:
    """
    Master loader. Call this from any script.
    Returns a dict with keys:
        results, wcm, wcs, standings, xg
    """
    print("\n=== Loading all datasets ===")
    results   = load_international_results()
    wcm, wcs  = load_wc_metadata()
    standings = load_wc_standings()
    xg        = load_statsbomb_xg()
    print("=== All datasets loaded ===\n")
    return {
        "results":   results,
        "wcm":       wcm,
        "wcs":       wcs,
        "standings": standings,
        "xg":        xg,
    }


if __name__ == "__main__":
    data = load_all_data()
    for k, v in data.items():
        print(f"  {k}: {len(v)} rows")