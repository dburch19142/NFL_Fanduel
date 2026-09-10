"""Pulls real per-game NFL player stats from the public nflverse project and
scores them using FanDuel's actual NFL classic scoring rules, to produce a
season-average projection for each offensive skill player.

FanDuel classic scoring (as used here):
  Passing:   0.04 pts/yard, +4/TD, -1/INT
  Rushing:   0.1 pts/yard,  +6/TD
  Receiving: 0.1 pts/yard,  +6/TD, +0.5/reception (half-PPR)
  Fumbles lost: -2
  Any 2-point conversion: +2

Team defense/special teams (DEF) scoring depends on points allowed, sacks,
takeaways and return TDs, which nflverse's weekly offense dataset doesn't
carry -- those projections fall back to FanDuel's own reported FPPG instead
(see player_pool.py).
"""
import time
import urllib.request

import pandas as pd

CACHE_DIR = "data/cache"
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # 1 day

PARQUET_URL = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{0}.parquet"

SUFFIXES = (" jr", " sr", " ii", " iii", " iv", " v")


def normalize_name(name: str) -> str:
    n = str(name).lower().strip()
    n = n.replace(".", "").replace("'", "").replace("-", " ")
    for suffix in SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return " ".join(n.split())


def _season_data_exists(season: int) -> bool:
    try:
        req = urllib.request.Request(PARQUET_URL.format(season), method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


def latest_available_season(start_year: int | None = None) -> int:
    """Finds the most recent season nflverse has published weekly data for."""
    if start_year is None:
        start_year = pd.Timestamp.now().year
    for year in range(start_year, start_year - 4, -1):
        if _season_data_exists(year):
            return year
    raise RuntimeError("Could not find any recent season of nflverse data.")


def _fanduel_points(df: pd.DataFrame) -> pd.Series:
    fumbles_lost = (
        df["sack_fumbles_lost"] + df["rushing_fumbles_lost"] + df["receiving_fumbles_lost"]
    )
    two_pt = (
        df["passing_2pt_conversions"]
        + df["rushing_2pt_conversions"]
        + df["receiving_2pt_conversions"]
    )
    return (
        df["passing_yards"] * 0.04
        + df["passing_tds"] * 4
        + df["interceptions"] * -1
        + df["rushing_yards"] * 0.1
        + df["rushing_tds"] * 6
        + df["receptions"] * 0.5
        + df["receiving_yards"] * 0.1
        + df["receiving_tds"] * 6
        + fumbles_lost * -2
        + two_pt * 2
    )


def _cache_path(season: int) -> str:
    return f"{CACHE_DIR}/offense_projections_{season}.csv"


def get_offense_projections(season: int | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Returns real season-average FanDuel-scored projections for offensive players.

    Columns: name, position, team, projected_points, games_played, season.
    """
    import os

    if season is None:
        season = latest_available_season()

    cache_file = _cache_path(season)
    if not force_refresh and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < CACHE_MAX_AGE_SECONDS:
            return pd.read_csv(cache_file)

    import nfl_data_py as nfl

    weekly = nfl.import_weekly_data([season])
    weekly = weekly[weekly["season_type"] == "REG"]
    weekly = weekly[weekly["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    weekly["fanduel_points"] = _fanduel_points(weekly)

    grouped = (
        weekly.groupby(["player_display_name", "position", "recent_team"])
        .agg(projected_points=("fanduel_points", "mean"), games_played=("fanduel_points", "count"))
        .reset_index()
        .rename(columns={"player_display_name": "name", "recent_team": "team"})
    )
    grouped["projected_points"] = grouped["projected_points"].round(2)
    grouped["season"] = season
    grouped["match_key"] = grouped["name"].apply(normalize_name)

    os.makedirs(CACHE_DIR, exist_ok=True)
    grouped.to_csv(cache_file, index=False)
    return grouped
