"""Unit tests for player_pool.py's matchup-eligibility filtering."""
import pandas as pd

import player_pool


def _sample_pool():
    return pd.DataFrame([
        {"name": "Eligible QB", "position": "QB", "team": "AAA", "salary": 8000, "projected_points": 20},
        {"name": "Ineligible QB", "position": "QB", "team": "BBB", "salary": 7500, "projected_points": 19},
        {"name": "Eligible WR", "position": "WR", "team": "AAA", "salary": 7000, "projected_points": 15},
        {"name": "Ineligible WR", "position": "WR", "team": "BBB", "salary": 6500, "projected_points": 14},
        {"name": "Some Defense", "position": "DEF", "team": "CCC", "salary": 3000, "projected_points": 8},
    ])


def test_matchup_eligibility_filters_out_ineligible_skill_players(monkeypatch):
    monkeypatch.setattr(
        player_pool,
        "build_eligibility",
        lambda *a, **k: {
            "season": 2026,
            "week": 2,
            "eligible": {
                "QB": {("eligible qb", "AAA")},
                "RB": set(),
                "WR": {("eligible wr", "AAA")},
                "TE": set(),
            },
            "defense": {"pass": set(), "rush": set()},
        },
    )

    filtered, note = player_pool._apply_matchup_eligibility(_sample_pool())

    assert set(filtered["name"]) == {"Eligible QB", "Eligible WR", "Some Defense"}
    assert "Ineligible QB" not in set(filtered["name"])
    assert "week 2" in note


def test_matchup_eligibility_defense_always_passes_through(monkeypatch):
    monkeypatch.setattr(
        player_pool,
        "build_eligibility",
        lambda *a, **k: {
            "season": 2026, "week": 2,
            "eligible": {"QB": set(), "RB": set(), "WR": set(), "TE": set()},
            "defense": {"pass": set(), "rush": set()},
        },
    )
    filtered, _ = player_pool._apply_matchup_eligibility(_sample_pool())
    assert "Some Defense" in set(filtered["name"])


def test_matchup_eligibility_falls_back_gracefully_when_unavailable(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(player_pool, "build_eligibility", _boom)

    pool = _sample_pool()
    filtered, note = player_pool._apply_matchup_eligibility(pool)

    assert len(filtered) == len(pool)  # unfiltered fallback, nobody dropped
    assert "unavailable" in note
