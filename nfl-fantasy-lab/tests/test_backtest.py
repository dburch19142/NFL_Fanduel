"""Unit tests for backtest mode (no network needed)."""
import sys
import types

import pandas as pd

import backtest
import matchup_filters
import nfl_data
import vegas_strategy

WEEKS = pd.DataFrame({"week": [5, 6, 7, 8], "season_type": "REG", "x": [1, 2, 3, 4]})


def _fake_parquet(monkeypatch, seen):
    def read_parquet(url, columns=None):
        seen.append(columns)
        return WEEKS if columns is None else WEEKS[columns]
    monkeypatch.setattr(backtest.pd, "read_parquet", read_parquet)


def test_live_reads_every_week(monkeypatch):
    monkeypatch.delenv("NFL_BACKTEST_WEEK", raising=False)
    _fake_parquet(monkeypatch, [])
    assert list(backtest.read_stats("u")["week"]) == [5, 6, 7, 8]
    assert backtest.describe() == "" and backtest.cache_dir("data/cache") == "data/cache"


def test_backtest_keeps_only_earlier_weeks(monkeypatch):
    monkeypatch.setenv("NFL_BACKTEST_WEEK", "7")
    seen = []
    _fake_parquet(monkeypatch, seen)
    assert list(backtest.read_stats("u")["week"]) == [5, 6]
    # A column list without "week" (play-by-play) still gets filtered, and
    # comes back as asked.
    df = backtest.read_stats("u", columns=["x"])
    assert seen[-1] == ["x", "week"]
    assert list(df.columns) == ["x"] and list(df["x"]) == [1, 2]


def test_backtest_caches_and_label(monkeypatch):
    monkeypatch.setenv("NFL_BACKTEST_WEEK", "7")
    monkeypatch.setenv("NFL_BACKTEST_SEASON", "2026")
    assert backtest.cache_dir("data/cache") == "data/cache/backtest_2026_w7"
    assert backtest.describe() == "BACKTEST of 2026 week 7 (stats through week 6)"
    assert matchup_filters.latest_season_with_player_data() == 2026


def test_backtest_week_is_the_backtest_week_played_or_not(monkeypatch):
    monkeypatch.setenv("NFL_BACKTEST_WEEK", "7")
    schedule = pd.DataFrame([
        {"week": 7, "game_type": "REG", "home_team": "BAL", "away_team": "CIN", "home_score": 13},
        {"week": 8, "game_type": "REG", "home_team": "BUF", "away_team": "BAL", "home_score": None},
    ])
    monkeypatch.setitem(sys.modules, "nfl_data_py", types.SimpleNamespace(import_schedules=lambda s: schedule))
    assert matchup_filters.get_upcoming_week_matchups(2026) == (7, {"BAL": "CIN", "CIN": "BAL"})
    monkeypatch.delenv("NFL_BACKTEST_WEEK")
    assert matchup_filters.get_upcoming_week_matchups(2026) == (8, {"BUF": "BAL", "BAL": "BUF"})


def test_backtest_skips_the_live_injury_report(monkeypatch):
    monkeypatch.setenv("NFL_BACKTEST_WEEK", "7")
    monkeypatch.setattr(nfl_data, "injury_report", lambda: (_ for _ in ()).throw(AssertionError("read")))
    pool = pd.DataFrame([{"name": "A", "position": "WR", "team": "CIN"}])
    result, note, removed = nfl_data.apply_injury_report(pool)
    assert result is pool and removed == []
    assert note == "live injury report skipped (backtest of week 7)"


def test_closing_odds_come_from_each_games_summary(monkeypatch):
    scoreboard = {"events": [{"id": "1", "date": "2026-10-25T17:00Z", "competitions": [{
        "odds": [],
        "competitors": [{"homeAway": "home", "team": {"abbreviation": "BAL"}},
                        {"homeAway": "away", "team": {"abbreviation": "CIN"}}],
    }]}]}
    monkeypatch.setattr(vegas_strategy, "fetch_json", lambda path, params: {
        "pickcenter": [{"overUnder": 46.5, "spread": 3.0}]} if path == "summary" and params == {"event": "1"} else {})
    odds = vegas_strategy.closing_odds(scoreboard)
    assert odds.to_dict("records") == [
        {"home": "BAL", "away": "CIN", "total": 46.5, "home_spread": 3.0, "kickoff": "2026-10-25T17:00Z"}]
    assert vegas_strategy.parse_scoreboard_odds(scoreboard).empty  # what the scoreboard alone gives
