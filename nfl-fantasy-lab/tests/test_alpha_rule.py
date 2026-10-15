"""Unit tests for the Alpha Rule strategy (no network needed)."""
import pandas as pd

import player_pool
from alpha_rule import afternoon_games, apply_alpha_rule
from optimizer import optimize_lineup

# Kickoffs are UTC; in October 17:00Z is 1pm ET, 20:25Z is 4:25pm ET and
# 00:20Z is the 8:20pm ET night game, which carries the highest total but
# is outside the 1pm and 4pm windows.
ODDS = pd.DataFrame([
    {"home": "JJJ", "away": "III", "total": 56.0, "home_spread": -3.0, "kickoff": "2026-10-05T00:20Z"},
    {"home": "BBB", "away": "AAA", "total": 52.0, "home_spread": -3.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "DDD", "away": "CCC", "total": 49.0, "home_spread": 9.0, "kickoff": "2026-10-04T20:25Z"},
    {"home": "FFF", "away": "EEE", "total": 48.0, "home_spread": -1.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "HHH", "away": "GGG", "total": 40.0, "home_spread": -6.0, "kickoff": "2026-10-04T20:05Z"},
])
TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ")
# FFF is the worst rush defense, EEE the second worst.
RUSH_ALLOWED = {"AAA": 100, "BBB": 105, "CCC": 110, "DDD": 115, "EEE": 140, "FFF": 150,
                "GGG": 95, "HHH": 90, "III": 200, "JJJ": 190}


def _player(name, pos, team, salary=5000, pts=10.0):
    return {"name": name, "position": pos, "team": team, "salary": salary, "projected_points": pts}


def _stat(name, pos, team, pass_attempts=0, share=0.0):
    return {"match_key": name.lower(), "position": pos, "team": team,
            "pass_attempts": pass_attempts, "target_share": share}


def _pool_and_stats():
    players, stats = [], []
    for team in TEAMS:
        players += [
            _player(f"{team} QB", "QB", team, 7000, 18),
            _player(f"{team} RB1", "RB", team, 6000, 12),
            _player(f"{team} RB2", "RB", team, 5000, 8),
            _player(f"{team} WR1", "WR", team, 7000, 14),
            _player(f"{team} WR2", "WR", team, 6000, 11),
            _player(f"{team} WR3", "WR", team, 4500, 6),
            _player(f"{team} TE", "TE", team, 4500, 7),
            _player(f"{team} Defense", "DEF", team, 3500, 7),
        ]
        stats += [
            _stat(f"{team} QB", "QB", team, pass_attempts=100),
            _stat(f"{team} WR1", "WR", team, share=0.27),
            _stat(f"{team} WR2", "WR", team, share=0.20),
            _stat(f"{team} WR3", "WR", team, share=0.10),
        ]
    return pd.DataFrame(players), pd.DataFrame(stats)


def _run(pool=None, stats=None, odds=ODDS):
    base_pool, base_stats = _pool_and_stats()
    return apply_alpha_rule(
        base_pool if pool is None else pool, odds, base_stats if stats is None else stats, RUSH_ALLOWED,
    )


def _group(pool, label):
    return set(pool.loc[pool["required_group"] == label, "name"])


def test_afternoon_games_keeps_only_the_1pm_and_4pm_windows():
    assert list(afternoon_games(ODDS)["home"]) == ["BBB", "DDD", "FFF", "HHH"]


def test_pool_restricted_to_1pm_and_4pm_games():
    pool, notes = _run()
    assert set(pool["team"]) == set(TEAMS) - {"III", "JJJ"}
    assert "AAA @ BBB (52)" in notes[0]
    assert "III @ JJJ" not in notes[0]


def test_qb_is_on_the_higher_implied_team_of_the_top_game():
    result, notes = _run()
    # BBB is a 3-point home favorite in the 52-point game: implied 27.5.
    assert _group(result, "QB") == {"BBB QB"}
    assert any("implied 27.5" in n for n in notes)


def test_wr1_stack_wr2_bring_back_and_wr3_from_the_second_game():
    result, _ = _run()
    assert _group(result, "WR1") == {"BBB WR1"}
    assert _group(result, "WR2") == {"AAA WR1"}
    # CCC is a 9-point road favorite in the second-highest over/under game.
    assert _group(result, "WR3") == {"CCC WR1"}


def test_wr1_is_the_team_leader_in_target_share_not_projection():
    pool, stats = _pool_and_stats()
    pool.loc[pool["name"] == "BBB WR2", "projected_points"] = 30
    result, _ = _run(pool, stats)
    assert _group(result, "WR1") == {"BBB WR1"}


def test_rbs_face_the_worst_and_second_worst_rush_defenses():
    result, notes = _run()
    assert _group(result, "RB1") == {"EEE RB1"}  # EEE plays FFF
    assert _group(result, "RB2") == {"FFF RB1"}  # FFF plays EEE
    assert any("second-worst rush D" in n for n in notes)


def test_rb_facing_a_team_whose_qb1_is_out_can_take_the_slot():
    pool, stats = _pool_and_stats()
    pool = pool[pool["name"] != "HHH QB"]
    pool = pd.concat([pool, pd.DataFrame([_player("HHH Backup", "QB", "HHH", 6000, 3)])])
    pool.loc[pool["name"] == "GGG RB1", "projected_points"] = 20  # GGG plays HHH
    result, notes = _run(pool, stats)
    assert any("QB1 out, starting a backup: HHH (HHH Backup)" in n for n in notes)
    assert _group(result, "RB1") == {"GGG RB1"}
    assert _group(result, "RB2") == {"FFF RB1"}
    assert any("faces a team whose QB1 is out" in n for n in notes)


def test_def_faces_the_lowest_implied_offense():
    result, notes = _run()
    # GGG is a 6-point underdog in the 40-point game: implied 17.0.
    assert _group(result, "DEF") == {"HHH Defense"}
    assert any("implied 17.0" in n for n in notes)


def test_def_skips_an_offense_already_stacked_in_the_lineup():
    odds = ODDS.copy()
    odds.loc[odds["home"] == "BBB", "home_spread"] = -30.0  # AAA implied 11.0, lowest on the slate
    result, notes = _run(odds=odds)
    assert _group(result, "WR2") == {"AAA WR1"}
    assert _group(result, "DEF") == {"HHH Defense"}
    assert any("not already in the lineup (AAA is)" in n for n in notes)


def test_wr3_skipped_with_only_one_afternoon_game():
    result, notes = _run(odds=ODDS[ODDS["home"].isin(["JJJ", "BBB"])])
    assert _group(result, "WR3") == set()
    assert any("WR3 rule skipped" in n for n in notes)


def test_optimizer_builds_a_lineup_from_the_alpha_picks():
    result, _ = _run()
    lineup = optimize_lineup(60000, result)
    assert set(result.loc[result["required_group"].notna(), "name"]) <= set(lineup["name"])


def test_alpha_strategy_falls_back_to_matchup_filters(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("odds unavailable")

    monkeypatch.setattr(player_pool, "build_alpha_pool", _boom)
    monkeypatch.setattr(player_pool, "build_eligibility", _boom)
    pool, _ = _pool_and_stats()
    result, note = player_pool._apply_alpha_strategy(pool)
    assert "Alpha Rule unavailable (odds unavailable)" in note
    assert len(result) == len(pool)
