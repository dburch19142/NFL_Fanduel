"""Backtest mode: replay a past slate as if it were that week's Sunday
morning.

Set NFL_BACKTEST_WEEK (and, for a past season, NFL_BACKTEST_SEASON) before
starting the app or a script. While it's set:

  * every nflverse stats read (player weeks, team weeks, play-by-play) goes
    through read_stats, which keeps only the weeks before the backtest
    week -- no result from the slate itself, or after it, leaks in;
  * caches live in their own folder per backtest season and week (see
    cache_dir), so live and backtest data never mix;
  * "this week" is the backtest week, with its full schedule, played or
    not (see matchup_filters.get_upcoming_week_matchups);
  * Vegas lines are the closing lines from each game's ESPN summary, since
    the scoreboard drops them once a game kicks off (see
    vegas_strategy.closing_odds);
  * the live injury report is skipped -- it would describe a later week.
    The uploaded FanDuel export's own injury designations still apply.

Unset (the default), everything works on live data as before.
"""
import os

import pandas as pd


def backtest_week() -> int | None:
    """The backtest week, or None when running live."""
    value = os.environ.get("NFL_BACKTEST_WEEK", "").strip()
    return int(value) if value else None


def backtest_season() -> int | None:
    """The backtest season, or None to use the latest season with data."""
    value = os.environ.get("NFL_BACKTEST_SEASON", "").strip()
    return int(value) if value else None


def describe() -> str:
    """One line for the pool source, empty when running live."""
    week = backtest_week()
    if week is None:
        return ""
    season = backtest_season()
    return f"BACKTEST of {f'{season} ' if season else ''}week {week} (stats through week {week - 1})"


def read_stats(url: str, columns: list[str] | None = None) -> pd.DataFrame:
    """pd.read_parquet for an nflverse stats file, keeping only the weeks
    before the backtest week when one is set."""
    week = backtest_week()
    if week is None:
        return pd.read_parquet(url, columns=columns)
    wanted = None if columns is None else list(dict.fromkeys(list(columns) + ["week"]))
    df = pd.read_parquet(url, columns=wanted)
    df = df[df["week"] < week]
    if columns is not None and "week" not in columns:
        df = df.drop(columns="week")
    return df


def cache_dir(base: str) -> str:
    """`base` when running live; a folder of its own per backtest season
    and week otherwise."""
    week = backtest_week()
    if week is None:
        return base
    return f"{base}/backtest_{backtest_season() or 'latest'}_w{week}"
