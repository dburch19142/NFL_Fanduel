"""Unit tests for the Beta strategy (beta_strategy.py and its pool filter in
player_pool.py), with made-up teams, lines and stats (no network)."""
import pandas as pd

import beta_strategy
import player_pool

# 2026-10-25 is a Sunday. Kickoffs in UTC: 17:00 = 1pm ET, 20:25 = 4:25pm ET,
# 00:20 the next day = Sunday night.
ODDS = pd.DataFrame([
    {"home": "RUN", "away": "SIV", "total": 45.0, "home_spread": -3.0, "kickoff": "2026-10-25T17:00Z"},
    {"home": "GRD", "away": "LEK", "total": 44.0, "home_spread": -1.0, "kickoff": "2026-10-25T20:25Z"},
    {"home": "NGT", "away": "HOL", "total": 47.0, "home_spread": -2.0, "kickoff": "2026-10-26T00:20Z"},
    {"home": "MID", "away": "AVG", "total": 41.0, "home_spread": -2.5, "kickoff": "2026-10-25T17:00Z"},
])


def _team_week():
    """A beats B 200-50 and C 180-60 on the ground; C vs D and B vs D fill in."""
    games = [  # (team, opponent, week, its rushing yards)
        ("A", "B", 1, 200), ("B", "A", 1, 50), ("C", "D", 1, 100), ("D", "C", 1, 120),
        ("A", "C", 2, 180), ("C", "A", 2, 60), ("B", "D", 2, 90), ("D", "B", 2, 150),
    ]
    return pd.DataFrame(games, columns=["team", "opponent_team", "week", "rushing_yards"])


def _ranks(offense_top, defense_bottom):
    """A ranks table with the named teams at the top of each list."""
    teams = ["RUN", "SIV", "GRD", "LEK", "NGT", "HOL", "MID", "AVG"]
    return pd.DataFrame({
        "rush_ypg": [100.0] * len(teams),
        "offense_rank": [offense_top.get(t, 20) for t in teams],
        "rush_allowed_ypg": [100.0] * len(teams),
        "defense_rank": [defense_bottom.get(t, 20) for t in teams],
    }, index=teams)


def test_rushing_ranks_rank_offense_and_defense_separately():
    ranks = beta_strategy.rushing_ranks(_team_week())
    # Offense: A 190, D 135, C 80, B 70 yards per game.
    assert ranks["offense_rank"].to_dict() == {"A": 1, "B": 4, "C": 3, "D": 2}
    # Allowed: B 175 (A's 200, D's 150), C 150, D 95, A 55.
    assert ranks["defense_rank"].to_dict() == {"A": 4, "B": 1, "C": 2, "D": 3}
    assert ranks.loc["B", "rush_allowed_ypg"] == 175


def test_qualifying_needs_top_offense_bad_defense_and_sunday_afternoon():
    ranks = _ranks(
        offense_top={"RUN": 1, "GRD": 2, "NGT": 3},
        defense_bottom={"SIV": 1, "HOL": 2},  # LEK isn't bottom 10
    )
    found = beta_strategy.qualifying_teams(ranks, ODDS)
    # GRD faces a good run D; NGT plays Sunday night; MID/AVG aren't top 10.
    assert [q["team"] for q in found] == ["RUN"]
    assert found[0]["opponent"] == "SIV" and found[0]["defense_rank"] == 1


def test_ties_at_the_cutoff_are_included():
    ranks = _ranks(offense_top={"RUN": 10, "GRD": 10}, defense_bottom={"SIV": 10, "LEK": 10})
    assert {q["team"] for q in beta_strategy.qualifying_teams(ranks, ODDS)} == {"RUN", "GRD"}


def _pool():
    rows = [
        ("Run Lead", "RB", "RUN", 8000, 18), ("Run Backup", "RB", "RUN", 5000, 8),
        ("Other Star", "RB", "MID", 9000, 25), ("Other Value", "RB", "AVG", 4500, 12),
        ("Good QB", "QB", "MID", 8000, 20),
        ("WR A", "WR", "MID", 7000, 15), ("WR B", "WR", "MID", 6500, 14), ("WR C", "WR", "AVG", 6000, 13),
        ("TE A", "TE", "AVG", 5000, 10),
        ("Some Defense", "DEF", "AVG", 3500, 8),
    ]
    return pd.DataFrame(rows, columns=["name", "position", "team", "salary", "projected_points"])


def _patch_live_data(monkeypatch, qualifying):
    everyone = {(n.lower(), t) for n, p, t in zip(_pool()["name"], _pool()["position"], _pool()["team"])}
    monkeypatch.setattr(player_pool, "build_eligibility", lambda *a, **k: {
        "season": 2026, "week": 7,
        "eligible": {pos: everyone for pos in ("QB", "RB", "WR", "TE")},
        "eligible_def_teams": {"AVG"},
        "matchups": {"RUN": "SIV", "SIV": "RUN", "MID": "AVG", "AVG": "MID"},
    })
    monkeypatch.setattr(player_pool, "_load_week_context", lambda *a, **k: {"odds": ODDS})
    monkeypatch.setattr(player_pool, "load_rushing_ranks", lambda season: None)
    monkeypatch.setattr(player_pool, "qualifying_teams", lambda ranks, odds: qualifying)


def test_beta_pool_keeps_only_qualifying_rbs(monkeypatch):
    _patch_live_data(monkeypatch, [{"team": "RUN", "opponent": "SIV", "offense_rank": 1, "rush_ypg": 180.0,
                                     "defense_rank": 1, "rush_allowed_ypg": 200.0}])
    pool, source = player_pool._apply_beta_strategy(_pool())

    assert set(pool.loc[pool["position"] == "RB", "name"]) == {"Run Lead", "Run Backup"}
    assert {"Good QB", "WR A", "TE A", "Some Defense"} <= set(pool["name"])
    assert "Beta strategy" in source
    assert any("RUN vs SIV" in n for n in pool.attrs["strategy_notes"])


def test_beta_falls_back_to_matchup_rbs_when_too_few(monkeypatch):
    _patch_live_data(monkeypatch, [])
    pool, _ = player_pool._apply_beta_strategy(_pool())

    assert "Other Star" in set(pool["name"])
    notes = pool.attrs["strategy_notes"]
    assert any("No team has" in n for n in notes)
    assert any("fall back" in n for n in notes)


def test_beta_unavailable_uses_matchup_filters(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(player_pool, "build_eligibility", _boom)
    pool, source = player_pool._apply_beta_strategy(_pool())
    assert "Beta strategy unavailable" in source
