"""Regenerates data/teams.csv from real NFL game results.

Run this any time to refresh the standings shown on the dashboard --
e.g. once a week during the season:

    python update_standings.py

Pulls the season's full schedule (with final scores) from the public
nflverse project via nfl_data_py, the same source real_stats.py uses for
player stats, and aggregates it into per-team win/loss/tie and points
for/against totals.
"""
import pandas as pd

TEAMS_CSV = "data/teams.csv"

TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def find_season_with_games(candidate_years):
    import nfl_data_py as nfl

    for year in candidate_years:
        schedule = nfl.import_schedules([year])
        played = schedule[schedule["game_type"] == "REG"].dropna(subset=["home_score", "away_score"])
        if len(played) > 0:
            return year, played
    raise RuntimeError(f"No completed games found for seasons {list(candidate_years)}.")


def build_standings(played_games: pd.DataFrame) -> pd.DataFrame:
    records = {
        abbr: {"wins": 0, "losses": 0, "ties": 0, "points_for": 0, "points_against": 0}
        for abbr in TEAM_NAMES
    }
    for _, game in played_games.iterrows():
        home, away = game["home_team"], game["away_team"]
        home_score, away_score = int(game["home_score"]), int(game["away_score"])

        records[home]["points_for"] += home_score
        records[home]["points_against"] += away_score
        records[away]["points_for"] += away_score
        records[away]["points_against"] += home_score

        if home_score > away_score:
            records[home]["wins"] += 1
            records[away]["losses"] += 1
        elif away_score > home_score:
            records[away]["wins"] += 1
            records[home]["losses"] += 1
        else:
            records[home]["ties"] += 1
            records[away]["ties"] += 1

    rows = [{"team": TEAM_NAMES[abbr], **stats} for abbr, stats in records.items()]
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def main():
    current_year = pd.Timestamp.now().year
    season, played_games = find_season_with_games([current_year, current_year - 1])

    weeks_seen = sorted(played_games["week"].unique())
    standings = build_standings(played_games)
    standings.to_csv(TEAMS_CSV, index=False)

    print(f"Season {season}, through week {weeks_seen[-1]} ({len(played_games)} completed games).")
    print(f"Wrote {len(standings)} teams to {TEAMS_CSV}.")


if __name__ == "__main__":
    main()
