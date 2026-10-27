"""The "Beta" lineup strategy: running backs come only from teams with a
top-10 rushing offense facing a bottom-10 rushing defense in a Sunday 1pm or
4pm ET game. Every other slot (QB, WR, TE, DEF, and a WR/TE FLEX) uses the
full-slate matchup filters, as in the matchup strategy (see
matchup_filters.py and player_pool.py).

  Rushing offense: rushing yards per game this regular season, all players.
  Rushing defense: rushing yards per game its opponents actually gained
        against it. Bottom 10 = the ten allowing the most.
  Ties at either cutoff are all included, so "top 10" can be eleven teams.

Every RB on a qualifying team stays in the pool and every other RB is
dropped, so both RB slots and an RB in FLEX all come from qualifying teams.
If the qualifying teams have fewer than two RBs on the (injury-filtered)
slate, the rule can't fill the roster: the RBs fall back to the matchup
filters, and the notes say so.
"""
import pandas as pd

from alpha_rule import afternoon_games
from backtest import read_stats
from matchup_filters import STATS_TEAM_WEEK_URL

TOP_N_RUSH_OFFENSE = 10
BOTTOM_N_RUSH_DEFENSE = 10


def rushing_ranks(team_week: pd.DataFrame) -> pd.DataFrame:
    """One row per team: rush_ypg and its rank (1 = most), rush_allowed_ypg
    and its rank (1 = most allowed, the worst defense). From nflverse team
    weekly stats (columns team, opponent_team, week, rushing_yards)."""
    games = team_week[["team", "opponent_team", "week", "rushing_yards"]]
    offense = games.groupby("team")["rushing_yards"].mean().rename("rush_ypg")
    defense = games.groupby("opponent_team")["rushing_yards"].mean().rename("rush_allowed_ypg")
    ranks = pd.concat([offense, defense], axis=1)
    ranks["offense_rank"] = ranks["rush_ypg"].rank(ascending=False, method="min").astype(int)
    ranks["defense_rank"] = ranks["rush_allowed_ypg"].rank(ascending=False, method="min").astype(int)
    return ranks


def qualifying_teams(ranks: pd.DataFrame, odds: pd.DataFrame) -> list[dict]:
    """Teams with a top-10 rushing offense playing a bottom-10 rushing defense
    in a Sunday 1pm or 4pm ET game (`odds` has home, away, kickoff)."""
    found = []
    for _, game in afternoon_games(odds).iterrows():
        for team, opponent in ((game["home"], game["away"]), (game["away"], game["home"])):
            if team not in ranks.index or opponent not in ranks.index:
                continue
            off, dfn = ranks.loc[team], ranks.loc[opponent]
            if off["offense_rank"] <= TOP_N_RUSH_OFFENSE and dfn["defense_rank"] <= BOTTOM_N_RUSH_DEFENSE:
                found.append({
                    "team": team,
                    "opponent": opponent,
                    "offense_rank": int(off["offense_rank"]),
                    "rush_ypg": round(float(off["rush_ypg"]), 1),
                    "defense_rank": int(dfn["defense_rank"]),
                    "rush_allowed_ypg": round(float(dfn["rush_allowed_ypg"]), 1),
                })
    return sorted(found, key=lambda q: q["offense_rank"])


def beta_rb_mask(pool: pd.DataFrame, teams: set[str]) -> pd.Series:
    """True for each RB on a qualifying team."""
    return (pool["position"] == "RB") & pool["team"].str.upper().isin(teams)


def beta_notes(qualifying: list[dict]) -> list[str]:
    if not qualifying:
        return ["No team has a top-10 rushing offense facing a bottom-10 rushing defense "
                "in a Sunday 1pm or 4pm game this week."]
    return [
        f"RB team {q['team']} vs {q['opponent']}: No. {q['offense_rank']} rushing offense "
        f"({q['rush_ypg']} yds/game) vs the No. {q['defense_rank']} worst rushing defense "
        f"({q['rush_allowed_ypg']} allowed/game)."
        for q in qualifying
    ]


def load_rushing_ranks(season: int) -> pd.DataFrame:
    team_week = read_stats(STATS_TEAM_WEEK_URL.format(season))
    return rushing_ranks(team_week[team_week["season_type"] == "REG"])
