"""Unit tests for the Charlie Rule strategy (no network needed)."""
import pandas as pd

import player_pool
from charlie_rule import apply_charlie_rule, defense_splash
from optimizer import optimize_lineup

# All Sunday 1pm ET (Oct 4, 2026). AAA is favored at BBB in a 47 total; DDD
# is favored over CCC in the highest total (52); EEE @ FFF is a 38 total;
# GGG @ HHH a 42.
ODDS = pd.DataFrame([
    {"home": "BBB", "away": "AAA", "total": 47.0, "home_spread": 3.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "DDD", "away": "CCC", "total": 52.0, "home_spread": -3.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "FFF", "away": "EEE", "total": 38.0, "home_spread": 2.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "HHH", "away": "GGG", "total": 42.0, "home_spread": -1.0, "kickoff": "2026-10-04T17:00Z"},
])
TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
SPLASH = {"GGG": 4.5, "HHH": 2.0, "EEE": 4.0, "CCC": 3.0}


def _player(name, pos, team, salary, pts=10.0):
    return {"name": name, "position": pos, "team": team, "salary": salary, "projected_points": pts}


def _stat(name, pos, team, share=0.0, carries=0, goal_line=0):
    return {"match_key": name.lower(), "position": pos, "team": team, "carries": carries,
            "target_share": share, "goal_line_targets": goal_line}


def _slate(overrides=None, stat_overrides=None):
    players, stats = [], []
    for t in TEAMS:
        players += [_player(f"{t} QB", "QB", t, 7000, 18), _player(f"{t} RB1", "RB", t, 7000, 12),
                    _player(f"{t} RB2", "RB", t, 4500, 5), _player(f"{t} WR1", "WR", t, 7000, 13),
                    _player(f"{t} WR2", "WR", t, 6000, 10), _player(f"{t} WR3", "WR", t, 4500, 5),
                    _player(f"{t} TE", "TE", t, 4500, 7), _player(f"{t} Defense", "DEF", t, 3800, 7)]
        stats += [_stat(f"{t} RB1", "RB", t, share=0.08, carries=120), _stat(f"{t} RB2", "RB", t, carries=60),
                  _stat(f"{t} WR1", "WR", t, share=0.26), _stat(f"{t} WR2", "WR", t, share=0.14),
                  _stat(f"{t} WR3", "WR", t, share=0.08), _stat(f"{t} TE", "TE", t, share=0.10)]
    pool, stats = pd.DataFrame(players), pd.DataFrame(stats)
    base = {
        # AAA: two 20%+ WRs. BBB's WR1 brings it back.
        ("AAA WR1", "salary"): 7000, ("AAA WR2", "salary"): 6500,
        # DDD: a pass-catching lead back and a punt TE with red-zone looks.
        ("DDD RB1", "salary"): 8000, ("DDD TE", "salary"): 4500,
        # FFF: a cheap workhorse; EEE's WR1 is the cheap run-back.
        ("FFF RB1", "salary"): 5500, ("EEE WR1", "salary"): 5500,
        # Cheap defenses.
        ("GGG Defense", "salary"): 3300, ("HHH Defense", "salary"): 3100, ("EEE Defense", "salary"): 3200,
    }
    stat_base = {
        ("aaa wr1", "target_share"): 0.24, ("aaa wr2", "target_share"): 0.22, ("bbb wr1", "target_share"): 0.30,
        ("ddd rb1", "target_share"): 0.21, ("ddd te", "goal_line_targets"): 1, ("ddd te", "target_share"): 0.12,
        ("fff rb1", "carries"): 240, ("eee wr1", "target_share"): 0.22, ("ddd wr2", "target_share"): 0.12,
    }
    for (name, col), value in {**base, **(overrides or {})}.items():
        pool.loc[pool["name"] == name, col] = value
    for (key, col), value in {**stat_base, **(stat_overrides or {})}.items():
        stats.loc[stats["match_key"] == key, col] = value
    return pool, stats


def _run(overrides=None, stat_overrides=None, bad_sack=("HHH", "FFF"), cap=60000):
    pool, stats = _slate(overrides, stat_overrides)
    return apply_charlie_rule(pool, ODDS, stats, set(bad_sack), SPLASH, salary_cap=cap)


def _group(pool, label):
    return set(pool.loc[pool["required_group"] == label, "name"])


def test_qb_is_on_a_favored_team_with_two_20_percent_wrs_not_the_top_total():
    result, notes = _run()
    assert _group(result, "QB") == {"AAA QB"}  # 47 total, while DDD's game is 52
    assert any(n.startswith("QB AAA QB") and "24% and 22%" in n for n in notes)


def test_double_stack_and_bring_back():
    result, _ = _run()
    assert _group(result, "WR1") == {"AAA WR1"} and _group(result, "FLEX") == {"AAA WR2"}
    assert _group(result, "WR2") == {"BBB WR1"}


def test_no_wr_duo_leaves_the_qb_to_the_optimizer():
    result, notes = _run(stat_overrides={("aaa wr2", "target_share"): 0.15})
    assert not _group(result, "QB") and not _group(result, "WR1")
    assert any("QB and the stack are left to the optimizer" in n for n in notes)


def test_bring_back_needs_25_percent():
    result, notes = _run(stat_overrides={("bbb wr1", "target_share"): 0.24})
    assert not _group(result, "WR2")
    assert any("No BBB WR1 at a 25%+ target share" in n for n in notes)


def test_rb1_is_a_pass_catching_lead_back_on_a_24_plus_team():
    result, _ = _run()
    assert _group(result, "RB1") == {"DDD RB1"}  # 67% of the carries, 21% target share, DDD 27.5


def test_rb2_is_a_cheap_workhorse_from_any_game():
    result, notes = _run()
    assert _group(result, "RB2") == {"FFF RB1"}  # 80% of the carries in a 38 total
    assert any("in EEE @ FFF (38)" in n for n in notes)


def test_wr3_is_the_run_back_from_rb2s_game():
    result, notes = _run()
    assert _group(result, "WR3") == {"EEE WR1"}
    assert any(n.startswith("WR3 EEE WR1") and "the run-back" in n for n in notes)


def test_wr3_falls_back_to_rb2s_teammate():
    result, notes = _run(stat_overrides={("eee wr1", "target_share"): 0.18, ("fff wr1", "target_share"): 0.25},
                         overrides={("FFF WR1", "salary"): 5800})
    assert _group(result, "WR3") == {"FFF WR1"}
    assert any("RB2's teammate" in n for n in notes)


def test_te_is_a_punt_with_red_zone_looks_on_a_26_plus_team():
    result, _ = _run()
    assert _group(result, "TE") == {"DDD TE"}


def test_def_is_cheap_facing_a_leaky_line_and_none_of_the_picks():
    # EEE's defense faces FFF (bottom-10 in sacks allowed) but FFF's RB1 is
    # the RB2 pick; GGG's faces HHH.
    result, _ = _run()
    assert _group(result, "DEF") == {"GGG Defense"}
    assert not ((result["team"] == "HHH") & (result["position"] != "DEF")).any()


def test_def_ties_go_to_more_splash_plays():
    result, _ = _run(bad_sack=("HHH", "GGG"))
    assert _group(result, "DEF") == {"GGG Defense"}  # 4.5 sacks + 2x INTs vs HHH's 2.0


def test_cap_moves_a_late_pick_to_the_next_candidate():
    # The picks come to $54,300 with GGG's $3,300 defense; HHH's is $3,100.
    result, notes = _run(bad_sack=("HHH", "GGG"), cap=54100)
    assert _group(result, "DEF") == {"HHH Defense"}
    assert any(n.startswith("DEF: GGG Defense would break") for n in notes)


def test_optimizer_builds_the_charlie_lineup_with_four_from_the_stack_game():
    result, _ = _run()
    lineup = optimize_lineup(60000, result)
    picks = set(result.loc[result["required_group"].notna(), "name"])
    assert picks <= set(lineup["name"])
    assert lineup["team"].isin(["AAA", "BBB"]).sum() == 4


def test_defense_splash_counts_two_per_interception():
    team_week = pd.DataFrame([
        {"team": "GGG", "season_type": "REG", "def_sacks": 3, "def_interceptions": 1},
        {"team": "GGG", "season_type": "REG", "def_sacks": 1, "def_interceptions": 0},
        {"team": "GGG", "season_type": "POST", "def_sacks": 9, "def_interceptions": 9},
    ])
    assert defense_splash(team_week) == {"GGG": 3.0}


def test_charlie_strategy_falls_back_to_matchup_filters(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("odds unavailable")

    monkeypatch.setattr(player_pool, "build_charlie_pool", _boom)
    monkeypatch.setattr(player_pool, "build_eligibility", _boom)
    pool, _ = _slate()
    result, note = player_pool._apply_charlie_strategy(pool)
    assert "Charlie Rule unavailable (odds unavailable)" in note
    assert len(result) == len(pool)
