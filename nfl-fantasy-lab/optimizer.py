"""Fantasy football lineup optimizer using integer linear programming.

Roster follows FanDuel's NFL classic contest rules:
1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DEF, $60,000 salary cap.
Maximizes total projected points subject to the salary cap.
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


class InfeasibleLineupError(Exception):
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


def optimize_lineup(salary_cap: float, players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the optimal lineup (as a DataFrame) under the given salary cap.

    Raises InfeasibleLineupError if no valid roster fits under the cap.
    """
    if players is None:
        players = load_players()
    players = _dedupe_players(players)

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

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleLineupError(
            f"No valid lineup fits under a ${salary_cap:,.0f} salary cap."
        )

    chosen = [i for i in players.index if picks[i].value() == 1]
    lineup = players.loc[chosen].sort_values(
        by="position", key=lambda col: col.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DEF": 4})
    ).reset_index(drop=True)
    return lineup


if __name__ == "__main__":
    lineup = optimize_lineup(DEFAULT_SALARY_CAP)
    print(lineup)
    print(f"\nTotal salary: ${lineup['salary'].sum():,.0f}")
    print(f"Total projected points: {lineup['projected_points'].sum():.1f}")
