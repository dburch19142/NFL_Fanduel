"""Bridge to the NFL Assistant's refreshed database (nfl-assistant/ in this repo):
live injury reports, depth charts, schedules, Vegas lines and season stats.

The lab uses it two ways:
  * apply_injury_report() drops players from the pool whose latest report or
    roster status says they won't play, before any strategy picks from it.
  * run_tool() lets the FanDuel assistant call the NFL Assistant's own tools.

Settings (environment):
  NFL_ASSISTANT_DIR   where nfl-assistant lives (default: ../nfl-assistant)
  NFL_DATABASE_URL    its database (default: sqlite:///<NFL_ASSISTANT_DIR>/nfl.db)

Refresh the data with `python refresh.py` inside nfl-assistant. If the package
or database is missing, everything here reports "unavailable" and the lab
carries on with just the FanDuel export's injury column.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from real_stats import normalize_name

HERE = Path(__file__).resolve().parent

# Any game designation counts, matching fanduel_import.OUT_INJURY_STATUSES:
# a clean bill of health is required.
EXCLUDED_DESIGNATIONS = {"Out", "Doubtful", "Questionable"}
# Roster statuses that mean the player can't suit up: reserve lists (IR, PUP,
# NFI, suspended), released, retired.
EXCLUDED_ROSTER_STATUSES = {"RES", "CUT", "RET"}

# An older report is ignored: by then the database's idea of the current week
# is wrong too, and last month's designations would cut healthy players.
MAX_REPORT_AGE = timedelta(days=7)

# FanDuel abbreviations that differ from nflverse's.
FANDUEL_TO_NFLVERSE = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS"}


class NflDataUnavailable(Exception):
    pass


def assistant_dir() -> Path:
    return Path(os.environ.get("NFL_ASSISTANT_DIR") or HERE.parent / "nfl-assistant").resolve()


def database_url() -> str:
    return os.environ.get("NFL_DATABASE_URL") or f"sqlite:///{(assistant_dir() / 'nfl.db').as_posix()}"


@lru_cache(maxsize=None)
def _session_factory(directory: str, url: str):
    if not (Path(directory) / "nfl_assistant").is_dir():
        raise NflDataUnavailable(f"NFL Assistant not found at {directory} (set NFL_ASSISTANT_DIR)")
    if url.startswith("sqlite:///") and not Path(url[len("sqlite:///"):]).exists():
        raise NflDataUnavailable(f"no NFL database at {url}; run refresh.py in nfl-assistant")
    if directory not in sys.path:
        sys.path.append(directory)
    try:
        from nfl_assistant.db import make_engine, make_session_factory
    except ImportError as exc:
        raise NflDataUnavailable(f"can't load the NFL Assistant ({exc}); pip install -r requirements.txt") from exc
    return make_session_factory(make_engine(url))


def _session():
    return _session_factory(str(assistant_dir()), database_url())()


def reset() -> None:
    """Forget the cached connection, e.g. after changing NFL_DATABASE_URL."""
    _session_factory.cache_clear()


def run_tool(name: str, tool_input: dict) -> dict:
    """Run one of the NFL Assistant's tools (get_injuries, get_depth_chart, ...)."""
    with _session() as session:
        from nfl_assistant.tools import execute_tool
        return execute_tool(session, name, tool_input)


def nfl_tool_schemas(names: list[str]) -> list[dict]:
    """The NFL Assistant's tool specs for `names`, or [] when it isn't installed."""
    try:
        _session_factory(str(assistant_dir()), database_url())
        from nfl_assistant.tools import TOOL_SCHEMAS
    except NflDataUnavailable:
        return []
    return [spec for spec in TOOL_SCHEMAS if spec["name"] in names]


def injury_report() -> dict:
    """Who is out this week, keyed by (normalized name, nflverse team).

    Uses each team's report only if it is for the current week; an older report
    (bye week, or this week's not out yet) says nothing about this week's game.
    Roster status (injured reserve and the like) is current whatever the week.
    """
    from sqlalchemy import func, select

    with _session() as session:
        from nfl_assistant.models import Injury, Player, TableMeta
        from nfl_assistant.refresh import current_week

        meta = session.get(TableMeta, "injuries")
        if meta is None:
            raise NflDataUnavailable("injury report never refreshed; run refresh.py in nfl-assistant")
        updated = meta.last_updated.strftime("%Y-%m-%d %H:%M UTC")
        # Stored as naive UTC by nfl-assistant.
        if datetime.now(timezone.utc).replace(tzinfo=None) - meta.last_updated > MAX_REPORT_AGE:
            raise NflDataUnavailable(f"last refreshed {updated}, too old to apply; run refresh.py in nfl-assistant")
        season = session.scalar(select(func.max(Player.season)))
        if season is None:
            raise NflDataUnavailable("NFL database is empty; run refresh.py in nfl-assistant")
        week = current_week(session, season)
        out: dict[tuple[str, str], str] = {}
        for p in session.scalars(
            select(Player).where(Player.season == season, Player.status.in_(EXCLUDED_ROSTER_STATUSES))
        ):
            if p.team:
                out[(normalize_name(p.full_name), p.team)] = f"roster status {p.status}"
        if week is not None:
            for row in session.scalars(
                select(Injury).where(
                    Injury.season == season, Injury.week == week, Injury.status.in_(EXCLUDED_DESIGNATIONS)
                )
            ):
                out[(normalize_name(row.player_name), row.team)] = f"{row.status} ({row.injury or 'undisclosed'})"
    return {"season": season, "week": week, "last_updated": updated, "out": out}


def apply_injury_report(pool: pd.DataFrame) -> tuple[pd.DataFrame, str, list[dict]]:
    """Drop pool players the latest report rules out. Defenses are kept.

    Returns (filtered pool, one-line note for the pool source, removed players).
    Leaves the pool untouched, with a note saying why, if the data isn't there.
    """
    try:
        report = injury_report()
    except Exception as exc:  # a missing or broken database shouldn't stop the optimizer
        return pool, f"live injury report unavailable ({exc})", []

    teams = pool["team"].str.upper().map(lambda t: FANDUEL_TO_NFLVERSE.get(t, t))
    reasons = [
        None if pos == "DEF" else report["out"].get((normalize_name(name), team))
        for name, pos, team in zip(pool["name"], pool["position"], teams)
    ]
    hit = pd.Series([r is not None for r in reasons], index=pool.index)
    removed = [
        {"name": n, "position": p, "team": t, "reason": r}
        for n, p, t, r in zip(pool["name"], pool["position"], pool["team"], reasons)
        if r is not None
    ]
    filtered = pool[~hit].reset_index(drop=True)
    filtered.attrs.update(pool.attrs)
    note = (
        f"live injury report (week {report['week']}, refreshed {report['last_updated']}): "
        f"{len(removed)} removed"
    )
    return filtered, note, removed
