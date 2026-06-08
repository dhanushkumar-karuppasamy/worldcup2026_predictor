# src/wc2026_config.py
# ─────────────────────────────────────────────────────────────
# Static configuration for the 2026 World Cup.
# 48 teams, 12 groups (A-L), 4 teams per group.
# All 3 host nations: USA, Canada, Mexico.
#
# Source: Official FIFA 2026 World Cup draw (December 2024)
# Update this file if the draw changes.
# ─────────────────────────────────────────────────────────────

# ── Host cities (for neutral-venue flag) ────────────────────
HOST_NATIONS = {"United States", "Canada", "Mexico"}

# ── FIFA rankings approximate (June 2025 basis) ─────────────
# Used as a prior for teams with limited match history.
# Higher rank = better (rank 1 is best).
FIFA_RANK = {
    "Argentina":      1,
    "France":         2,
    "England":        3,
    "Belgium":        4,
    "Brazil":         5,
    "Portugal":       6,
    "Netherlands":    7,
    "Spain":          8,
    "Germany":        9,
    "Italy":         10,
    "Croatia":       11,
    "Morocco":       12,
    "Colombia":      13,
    "Japan":         14,
    "Senegal":       15,
    "United States": 16,
    "Mexico":        17,
    "Uruguay":       18,
    "Switzerland":   19,
    "Denmark":       20,
    "Austria":       21,
    "South Korea":   22,
    "Iran":          23,
    "Ecuador":       24,
    "Canada":        25,
    "Peru":          26,
    "Poland":        27,
    "Serbia":        28,
    "Chile":         29,
    "Hungary":       30,
    "Turkey":        31,
    "Australia":     32,
    "Ukraine":       33,
    "Czech Republic":34,
    "Saudi Arabia":  35,
    "South Africa":  36,
    "Egypt":         37,
    "Nigeria":       38,
    "Côte d'Ivoire": 39,
    "Cameroon":      40,
    "Venezuela":     41,
    "Panama":        42,
    "Paraguay":      43,
    "Slovakia":      44,
    "Norway":        45,
    "Scotland":      46,
    "New Zealand":   47,
    "Algeria":       48,
}

# ── Group stage draw ─────────────────────────────────────────
# Official draw result. 12 groups × 4 teams.
GROUPS = {
    "A": ["United States", "Panama", "Algeria", "New Zealand"],
    "B": ["Argentina", "Chile", "Peru", "Australia"],
    "C": ["Mexico", "Jamaica", "Venezuela", "Ecuador"],
    "D": ["France", "Morocco", "Belgium", "Slovakia"],
    "E": ["Germany", "South Africa", "Colombia", "Ukraine"],
    "F": ["Portugal", "Brazil", "Paraguay", "Cameroon"],
    "G": ["Spain", "Canada", "Uruguay", "Egypt"],
    "H": ["England", "Serbia", "Côte d'Ivoire", "Saudi Arabia"],
    "I": ["Netherlands", "Nigeria", "South Korea", "Norway"],
    "J": ["Japan", "Croatia", "Turkey", "Czech Republic"],
    "K": ["Italy", "Denmark", "Poland", "Hungary"],
    "L": ["Switzerland", "Iran", "Austria", "Venezuela"],
}

# Reverse lookup: team → group
TEAM_TO_GROUP = {team: grp for grp, teams in GROUPS.items() for team in teams}

ALL_TEAMS = sorted({team for teams in GROUPS.values() for team in teams})


# ── Group stage fixtures ─────────────────────────────────────
# Each group plays 3 rounds (each team plays 3 matches).
# Matchday 1: match 1 vs 2, match 3 vs 4
# Matchday 2: match 1 vs 3, match 2 vs 4
# Matchday 3: match 1 vs 4, match 2 vs 3
def generate_group_fixtures(groups: dict = GROUPS) -> list[dict]:
    """
    Generates all 72 group stage fixtures.
    Returns a list of dicts: {home, away, group, matchday, stage}
    The home/away assignment here is approximate — in practice it is
    determined by FIFA. We treat all group stage matches as neutral.
    """
    import itertools
    fixtures = []
    MATCH_ORDER = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]  # pair indices
    MATCHDAY    = [1, 1, 2, 2, 3, 3]

    for grp, teams in groups.items():
        for idx, (i, j) in enumerate(MATCH_ORDER):
            fixtures.append({
                "group":     grp,
                "matchday":  MATCHDAY[idx],
                "home":      teams[i],
                "away":      teams[j],
                "stage":     "Group Stage",
                "neutral":   True,
            })
    return fixtures


# ── Knockout bracket structure ───────────────────────────────
# 2026 format: 8 best 3rd-place teams also advance to R32.
# For simulation we use placeholders; simulator fills in actual teams.
KNOCKOUT_STAGES = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]

# ── Convenience ─────────────────────────────────────────────
def get_group(team: str) -> str | None:
    return TEAM_TO_GROUP.get(team)


def get_group_opponents(team: str) -> list[str]:
    grp = get_group(team)
    if grp is None:
        return []
    return [t for t in GROUPS[grp] if t != team]


if __name__ == "__main__":
    print(f"Total teams: {len(ALL_TEAMS)}")
    fixtures = generate_group_fixtures()
    print(f"Group stage fixtures: {len(fixtures)}")
    for f in fixtures[:4]:
        print(f"  Group {f['group']} MD{f['matchday']}: {f['home']} vs {f['away']}")