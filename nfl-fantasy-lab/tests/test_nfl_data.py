"""Tests for the bridge to the NFL Assistant's database (nfl_data.py), against a
small database built with nfl-assistant's own schema. Skipped when nfl-assistant
isn't installed alongside (set NFL_ASSISTANT_DIR to point at it)."""
import json
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

import nfl_data
from fantasy_assistant import FantasyAssistant, FantasyTools
from optimizer import PLAYERS_CSV

if not (nfl_data.assistant_dir() / "nfl_assistant").is_dir():
    pytest.skip(f"nfl-assistant not found at {nfl_data.assistant_dir()}", allow_module_level=True)
sys.path.append(str(nfl_data.assistant_dir()))
pytest.importorskip("sqlalchemy")

from nfl_assistant.db import init_db, make_engine, make_session_factory  # noqa: E402
from nfl_assistant.models import Game, Injury, Player, TableMeta, Team  # noqa: E402

SEASON = 2026


def _hours_ago(hours):
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)


def _player(pid, name, team, pos, status="ACT"):
    return Player(player_id=pid, full_name=name, name_key=name.lower(), team=team,
                  position=pos, status=status, season=SEASON)


def _injury(name, team, pos, status, week=8, injury="Hamstring"):
    return Injury(season=SEASON, week=week, team=team, player_name=name, position=pos,
                  status=status, injury=injury)


@pytest.fixture
def nfl_db(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'nfl.db').as_posix()}"
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        s.add_all([
            Team(abbr="DEN", name="Denver Broncos", nickname="Broncos"),
            Team(abbr="SEA", name="Seattle Seahawks", nickname="Seahawks"),
            Team(abbr="JAX", name="Jacksonville Jaguars", nickname="Jaguars"),
            # Week 7 is played, week 8 is not, so week 8 is the current week.
            Game(game_id="g7", season=SEASON, week=7, away_team="SEA", home_team="DEN",
                 away_score=20, home_score=17, gameday=date(2026, 10, 18)),
            Game(game_id="g8", season=SEASON, week=8, away_team="DEN", home_team="SEA",
                 gameday=date(2026, 10, 25), total_line=44.5, spread_line=2.5),
            _player("p1", "Courtland Sutton", "DEN", "WR"),
            _player("p2", "Cooper Kupp", "SEA", "WR"),
            _player("p3", "Evan Engram", "DEN", "TE", status="RES"),
            _player("p4", "Bo Nix", "DEN", "QB"),
            _player("p5", "Travis Hunter", "JAX", "WR"),
            _injury("Courtland Sutton", "DEN", "WR", "Out"),
            _injury("Cooper Kupp", "SEA", "WR", "Questionable", injury="Ankle"),
            _injury("Bo Nix", "DEN", "QB", None),  # practice report only
            _injury("Jaylen Waddle", "DEN", "WR", "Out", week=7),  # last week's report
            _injury("Travis Hunter", "JAX", "WR", "Doubtful"),
            TableMeta(table_name="injuries", last_updated=_hours_ago(3),
                      row_count=5, source="test"),
        ])
        s.commit()
    monkeypatch.setenv("NFL_DATABASE_URL", url)
    nfl_data.reset()
    yield url
    nfl_data.reset()


def sample_with_injuries(strategy="matchup"):
    pool, note, removed = nfl_data.apply_injury_report(pd.read_csv(PLAYERS_CSV))
    pool.attrs["injury_exclusions"] = removed
    return pool, f"sample data + {note}"


def test_report_removes_designated_and_reserve_players(nfl_db):
    pool = pd.DataFrame([
        {"name": "Courtland Sutton", "position": "WR", "team": "DEN", "salary": 8800, "projected_points": 9},
        {"name": "Cooper Kupp", "position": "WR", "team": "SEA", "salary": 7600, "projected_points": 8},
        {"name": "Evan Engram", "position": "TE", "team": "DEN", "salary": 5600, "projected_points": 5},
        {"name": "Bo Nix", "position": "QB", "team": "DEN", "salary": 12000, "projected_points": 14},
        {"name": "Jaylen Waddle", "position": "WR", "team": "DEN", "salary": 10000, "projected_points": 8},
        {"name": "Travis Hunter", "position": "WR", "team": "JAC", "salary": 6000, "projected_points": 7},
        {"name": "Denver Broncos", "position": "DEF", "team": "DEN", "salary": 6800, "projected_points": 8},
    ])
    kept, note, removed = nfl_data.apply_injury_report(pool)

    # Out, Questionable, injured reserve, and Doubtful under FanDuel's JAC code.
    assert {r["name"] for r in removed} == {"Courtland Sutton", "Cooper Kupp", "Evan Engram", "Travis Hunter"}
    # No designation, last week's report, and defenses all stay.
    assert set(kept["name"]) == {"Bo Nix", "Jaylen Waddle", "Denver Broncos"}
    assert "week 8" in note and "4 removed" in note
    assert next(r for r in removed if r["name"] == "Cooper Kupp")["reason"] == "Questionable (Ankle)"


def test_missing_database_leaves_pool_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("NFL_DATABASE_URL", f"sqlite:///{(tmp_path / 'none.db').as_posix()}")
    nfl_data.reset()
    pool = pd.read_csv(PLAYERS_CSV)
    kept, note, removed = nfl_data.apply_injury_report(pool)
    assert len(kept) == len(pool) and removed == []
    assert "unavailable" in note
    nfl_data.reset()


def test_stale_report_is_not_applied(nfl_db):
    from nfl_assistant.db import make_engine, make_session_factory as factory

    with factory(make_engine(nfl_db))() as s:
        s.get(TableMeta, "injuries").last_updated = _hours_ago(24 * 8)
        s.commit()
    pool = pd.DataFrame([{"name": "Courtland Sutton", "position": "WR", "team": "DEN",
                          "salary": 8800, "projected_points": 9}])
    kept, note, removed = nfl_data.apply_injury_report(pool)
    assert len(kept) == 1 and removed == []
    assert "too old to apply" in note


def test_optimizer_never_picks_an_out_player(nfl_db):
    tools = FantasyTools(pool_builder=sample_with_injuries)
    result = tools.run("build_lineups", {"strategy": "matchup", "count": 5})
    picked = {p["name"] for lineup in result["lineups"] for p in lineup["players"]}
    assert result["lineups"]
    assert not picked & {"Courtland Sutton", "Cooper Kupp", "Evan Engram"}
    assert {r["name"] for r in result["removed_by_injury_report"]} >= {"Courtland Sutton", "Cooper Kupp"}


def test_find_player_explains_an_injury_removal(nfl_db):
    result = FantasyTools(pool_builder=sample_with_injuries).run("find_player", {"name": "Courtland Sutton"})
    assert result["removed_by_injury_report"]["reason"].startswith("Out")


def test_assistant_offers_and_runs_nfl_tools(nfl_db):
    tools = FantasyTools(pool_builder=sample_with_injuries)
    names = {spec["name"] for spec in FantasyAssistant(client=object(), tools=tools).tool_specs}
    assert {"get_injuries", "get_depth_chart", "get_player", "build_lineups"} <= names

    report = tools.run("get_injuries", {"team": "Broncos"})
    assert report["team"] == "DEN" and report["report_week"] == 8
    assert json.dumps(report, default=str)  # serializable for the tool_result
