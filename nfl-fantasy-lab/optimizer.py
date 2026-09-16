"""Fantasy football lineup optimizer using integer linear programming.

Roster follows FanDuel's NFL classic contest rules:
1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DEF, $60,000 salary cap.
Maximizes total projected points subject to the salary cap.

Also enforces a QB stack: whichever QB is picked must be paired with at
least one WR or TE from the same team, since a passing touchdown scores
for both the QB and his pass-catcher -- a standard DFS strategy.
"""
import pandas as pd
import pulp

from real_stats import normalize_name

PLAYERS_CSV = "data/players.csv"
DEFAULT_SALARY_CAP = 60000

ROSTER_SLOTS = {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 1,
    "DEF": 1,
}
FLEX_ELIGIBLE = ("RB", "WR", "TE")
FLEX_COUNT = 1
STACK_POSITIONS = ("WR", "TE")


class InfeasibleLineupError(Exception):
    pass


class PlayerNotFoundError(Exception):
    pass


def load_players(path: str = PLAYERS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def _dedupe_players(players: pd.DataFrame) -> pd.DataFrame:
    """Collapses rows that represent the same real person into one.

    Keyed on normalized name + team (not name alone) so two different real
    players who happen to share a name -- e.g. the two NFL WRs both named
    Mike Williams -- aren't merged together, since they never share a team.
    Keeps the highest-scoring row of each duplicate group.
    """
    keys = players["name"].apply(normalize_name) + "|" + players["team"].astype(str).str.upper()
    deduped = (
        players.assign(_dedupe_key=keys)
        .sort_values("projected_points", ascending=False)
        .drop_duplicates(subset="_dedupe_key", keep="first")
        .drop(columns="_dedupe_key")
    )
    return deduped.reset_index(drop=True)


def _solve(
    players: pd.DataFrame,
    salary_cap: float,
    forbidden_combos: list[frozenset[int]],
    required_index: int | None = None,
) -> list[int]:
    """Solves one lineup, forbidding any combination of picks that exactly
    matches a previously-found lineup (by row index into `players`), and
    optionally forcing a specific row index to be included (a "must-include"
    or "locked" player, in DFS terms).

    Returns the list of chosen row indices. Raises InfeasibleLineupError if
    no valid, not-yet-used roster fits under the cap.
    """
    prob = pulp.LpProblem("lineup_optimizer", pulp.LpMaximize)
    picks = {i: pulp.LpVariable(f"pick_{i}", cat="Binary") for i in players.index}

    # Objective: maximize total projected points.
    prob += pulp.lpSum(picks[i] * players.loc[i, "projected_points"] for i in players.index)

    # Salary cap constraint.
    prob += pulp.lpSum(picks[i] * players.loc[i, "salary"] for i in players.index) <= salary_cap

    # Exact roster size.
    total_slots = sum(ROSTER_SLOTS.values()) + FLEX_COUNT
    prob += pulp.lpSum(picks.values()) == total_slots

    # Fixed position minimums (QB and DEF are exact; RB/WR/TE get their base count
    # plus whatever the FLEX slot decides to add).
    for pos in ("QB", "DEF"):
        idx = players.index[players["position"] == pos]
        prob += pulp.lpSum(picks[i] for i in idx) == ROSTER_SLOTS[pos]

    for pos in FLEX_ELIGIBLE:
        idx = players.index[players["position"] == pos]
        prob += pulp.lpSum(picks[i] for i in idx) >= ROSTER_SLOTS[pos]

    # Cap RB/WR/TE combined at base + FLEX so the solver can't stack extra flex slots.
    flex_idx = players.index[players["position"].isin(FLEX_ELIGIBLE)]
    prob += pulp.lpSum(picks[i] for i in flex_idx) == sum(
        ROSTER_SLOTS[p] for p in FLEX_ELIGIBLE
    ) + FLEX_COUNT

    # QB stack: if a given QB is picked, at least one WR/TE from his team
    # must be picked too. picks[q] <= sum(teammates) forces the sum to be
    # >=1 whenever picks[q]=1, and is a no-op whenever picks[q]=0.
    qb_idx = players.index[players["position"] == "QB"]
    for q in qb_idx:
        team = players.loc[q, "team"]
        teammates = players.index[
            (players["team"] == team) & (players["position"].isin(STACK_POSITIONS))
        ]
        prob += picks[q] <= pulp.lpSum(picks[i] for i in teammates)

    # Uniqueness: a lineup that picks every single player from an earlier
    # lineup is forbidden, forcing at least one swap versus each one already found.
    for combo in forbidden_combos:
        prob += pulp.lpSum(picks[i] for i in combo) <= len(combo) - 1

    if required_index is not None:
        prob += picks[required_index] == 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleLineupError(
            f"No valid lineup fits under a ${salary_cap:,.0f} salary cap."
        )

    return [i for i in players.index if picks[i].value() == 1]


def _to_lineup_df(players: pd.DataFrame, chosen: list[int]) -> pd.DataFrame:
    return players.loc[chosen].sort_values(
        by="position", key=lambda col: col.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DEF": 4})
    ).reset_index(drop=True)


def _resolve_required_player(
    players: pd.DataFrame, name: str, team: str | None = None
) -> int:
    """Finds the row index of a must-include player by (normalized) name,
    optionally narrowed by team when the name alone is ambiguous.

    Raises PlayerNotFoundError with the closest name matches if there's no
    match, or more than one and no team was given to disambiguate.
    """
    target = normalize_name(name)
    keys = players["name"].apply(normalize_name)
    matches = players.index[keys == target]

    if team:
        team_matches = matches[players.loc[matches, "team"].str.upper() == team.upper()]
        if len(team_matches) > 0:
            matches = team_matches

    if len(matches) == 0:
        close = players[keys.str.contains(target.split(" ")[-1], na=False)]["name"].unique()
        hint = f" Close matches in the pool: {list(close)[:5]}." if len(close) else ""
        raise PlayerNotFoundError(f"No player named '{name}' found in the pool.{hint}")

    if len(matches) > 1:
        teams = players.loc[matches, "team"].tolist()
        raise PlayerNotFoundError(
            f"Multiple players named '{name}' found (teams: {teams}). "
            "Specify a team to disambiguate."
        )

    return matches[0]


def optimize_lineup(salary_cap: float, players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the single optimal lineup (as a DataFrame) under the given salary cap.

    Raises InfeasibleLineupError if no valid roster fits under the cap.
    """
    if players is None:
        players = load_players()
    players = _dedupe_players(players)
    chosen = _solve(players, salary_cap, forbidden_combos=[])
    return _to_lineup_df(players, chosen)


def optimize_lineups(
    salary_cap: float,
    players: pd.DataFrame | None = None,
    count: int = 5,
    required_name: str | None = None,
    required_team: str | None = None,
) -> list[pd.DataFrame]:
    """Returns up to `count` distinct lineups, best first.

    Each lineup differs from every earlier one by at least one player.
    Later lineups trade off some points for that variety, since each one
    is the best lineup that avoids exactly repeating any earlier one.
    Raises InfeasibleLineupError if not even one lineup fits under the cap;
    returns fewer than `count` lineups if the pool is too thin to build more
    distinct ones (e.g. FLEX-eligible depth at one position runs out).

    If `required_name` is given, every returned lineup is forced to include
    that player (a "locked" pick, in DFS terms) -- e.g. a QB you've decided
    on for a favorable matchup, letting the solver build the best roster
    around them. Raises PlayerNotFoundError if no such player is in the pool.
    """
    if players is None:
        players = load_players()
    players = _dedupe_players(players)

    required_index = None
    if required_name is not None:
        required_index = _resolve_required_player(players, required_name, required_team)

    lineups: list[pd.DataFrame] = []
    forbidden_combos: list[frozenset[int]] = []
    for _ in range(count):
        try:
            chosen = _solve(players, salary_cap, forbidden_combos, required_index)
        except InfeasibleLineupError:
            break
        lineups.append(_to_lineup_df(players, chosen))
        forbidden_combos.append(frozenset(chosen))

    if not lineups:
        raise InfeasibleLineupError(
            f"No valid lineup fits under a ${salary_cap:,.0f} salary cap."
        )
    return lineups


if __name__ == "__main__":
    for n, lineup in enumerate(optimize_lineups(DEFAULT_SALARY_CAP), start=1):
        print(f"=== Lineup {n} ===")
        print(lineup)
        print(f"Total salary: ${lineup['salary'].sum():,.0f}")
        print(f"Total projected points: {lineup['projected_points'].sum():.1f}\n")
