"""Unit tests for the Alpha Rule strategy (no network needed)."""
import pandas as pd

import player_pool
from alpha_rule import afternoon_games, apply_alpha_rule, defense_vs_receivers
from optimizer import optimize_lineup

# Kickoffs are UTC; in October 17:00Z is 1pm ET, 20:25Z is 4:25pm ET and
# 00:20Z is the 8:20pm ET night game, which carries the highest total but
# is outside the 1pm and 4pm windows. Oct 4, 2026 is a Sunday; LLL hosts a
# Saturday 1pm game. HHH-GGG is a Sunday afternoon game, but only the fourth
# highest total among them, so it's left out too.
ODDS = pd.DataFrame([
    {"home": "JJJ", "away": "III", "total": 56.0, "home_spread": -3.0, "kickoff": "2026-10-05T00:20Z"},
    {"home": "LLL", "away": "KKK", "total": 58.0, "home_spread": -3.0, "kickoff": "2026-10-03T17:00Z"},
    {"home": "BBB", "away": "AAA", "total": 52.0, "home_spread": -3.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "DDD", "away": "CCC", "total": 49.0, "home_spread": 9.0, "kickoff": "2026-10-04T20:25Z"},
    {"home": "FFF", "away": "EEE", "total": 48.0, "home_spread": -1.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "HHH", "away": "GGG", "total": 40.0, "home_spread": -6.0, "kickoff": "2026-10-04T20:05Z"},
])
TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL")
ALPHA_TEAMS = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}
# FFF is the worst rush defense in the top-3 games, EEE the second worst.
RUSH_ALLOWED = {"AAA": 100, "BBB": 105, "CCC": 110, "DDD": 115, "EEE": 140, "FFF": 150,
                "GGG": 160, "HHH": 170, "III": 200, "JJJ": 190, "KKK": 210, "LLL": 220}
# 22 other teams with stingier pass defenses, so none of the slate's teams
# is a top-10 pass D unless a test makes it one.
PASS_ALLOWED = {**{t: 250.0 for t in TEAMS}, **{f"T{n:02}": 180.0 for n in range(22)}}


def _player(name, pos, team, salary=5000, pts=10.0):
    return {"name": name, "position": pos, "team": team, "salary": salary, "projected_points": pts}


def _stat(name, pos, team, pass_attempts=0, share=0.0, carries=0, goal_line=0):
    return {"match_key": name.lower(), "position": pos, "team": team, "pass_attempts": pass_attempts,
            "carries": carries, "target_share": share, "goal_line_targets": goal_line}


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


def _run(pool=None, stats=None, odds=ODDS, pass_allowed=PASS_ALLOWED, defense_vs=None):
    base_pool, base_stats = _pool_and_stats()
    return apply_alpha_rule(
        base_pool if pool is None else pool, odds, base_stats if stats is None else stats,
        RUSH_ALLOWED, pass_allowed, defense_vs,
    )


def _group(pool, label):
    return set(pool.loc[pool["required_group"] == label, "name"])


def test_afternoon_games_keeps_only_sunday_1pm_and_4pm_windows():
    assert list(afternoon_games(ODDS)["home"]) == ["BBB", "DDD", "FFF", "HHH"]


def test_pool_restricted_to_the_top_3_sunday_afternoon_games():
    pool, notes = _run()
    assert set(pool["team"]) == ALPHA_TEAMS
    assert "Top 3 Sunday" in notes[0]
    assert "AAA @ BBB (52), CCC @ DDD (49), EEE @ FFF (48)" in notes[0]
    for left_out in ("III @ JJJ", "KKK @ LLL", "GGG @ HHH"):
        assert left_out not in notes[0]


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


def test_rbs_are_on_favored_teams_ranked_by_the_weaker_rush_d():
    # EEE faces the worst rush D in the top 3 (FFF), but EEE is an underdog
    # implied for 23.5, so its back is passed over. GGG and HHH allow more
    # rush yards, but their game isn't in the top 3.
    result, notes = _run()
    assert _group(result, "RB1") == {"FFF RB1"}  # FFF plays EEE, the second-worst rush D
    assert _group(result, "RB2") == {"CCC RB1"}  # CCC plays DDD, the next worst
    assert any("second-worst rush D" in n for n in notes)


def test_rb_with_no_favored_team_considers_every_lead_back():
    odds = ODDS.copy()
    # Every top-3 game a pick'em at 45: each team implied 22.5, none favored.
    odds.loc[odds["home"].isin(["BBB", "DDD", "FFF"]), ["total", "home_spread"]] = [45.0, 0.0]
    odds.loc[odds["home"] == "BBB", "total"] = 46.0  # keep BBB-AAA the top game
    result, notes = _run(odds=odds)
    assert any("considers every lead back" in n for n in notes)
    assert _group(result, "RB1") == {"EEE RB1"}  # faces FFF, the worst rush D


def test_rb_facing_a_team_whose_qb1_is_out_can_take_the_slot():
    pool, stats = _pool_and_stats()
    pool = pool[pool["name"] != "DDD QB"]
    pool = pd.concat([pool, pd.DataFrame([_player("DDD Backup", "QB", "DDD", 6000, 3)])])
    result, notes = _run(pool, stats)
    assert any("QB1 out, starting a backup: DDD (DDD Backup)" in n for n in notes)
    # CCC plays DDD: facing a team down its QB1 outranks the weaker rush D.
    assert _group(result, "RB1") == {"CCC RB1"}
    assert _group(result, "RB2") == {"FFF RB1"}
    assert any("whose QB1 is out" in n for n in notes)


def test_rb_facing_a_team_whose_wr1_is_out_can_take_the_slot():
    pool, stats = _pool_and_stats()
    pool = pool[pool["name"] != "DDD WR1"]
    result, notes = _run(pool, stats)
    assert any("WR1 out: DDD" in n for n in notes)
    assert _group(result, "RB1") == {"CCC RB1"}
    assert any("whose WR1 is out" in n for n in notes)


def test_rb1_is_the_fill_in_for_an_injured_lead_back():
    pool, stats = _pool_and_stats()
    pool = pool[pool["name"] != "DDD RB1"]
    stats = pd.concat([stats, pd.DataFrame([
        _stat("DDD RB1", "RB", "DDD", carries=100),
        _stat("DDD RB2", "RB", "DDD", carries=20),
    ])])
    result, notes = _run(pool, stats)
    assert any("Lead back out: DDD" in n for n in notes)
    assert _group(result, "RB1") == {"DDD RB2"}
    assert any("fills in for DDD's injured lead back" in n for n in notes)
    # RB2: FFF's back, the favored back facing the weakest rush D (EEE).
    assert _group(result, "RB2") == {"FFF RB1"}


def test_wr_facing_a_top_10_pass_d_swaps_to_his_teams_wr2():
    # AAA becomes a top-10 pass D, so the QB's WR1 (BBB, playing AAA) swaps.
    result, notes = _run(pass_allowed={**PASS_ALLOWED, "AAA": 150.0})
    assert _group(result, "WR1") == {"BBB WR2"}
    assert _group(result, "WR2") == {"AAA WR1"}
    assert _group(result, "WR3") == {"CCC WR1"}
    assert any("swapped to BBB's WR2 since WR1 BBB WR1 faces AAA, a top-10 pass D" in n for n in notes)


def test_wr_swap_applies_to_the_bring_back_but_not_wr3():
    result, _ = _run(pass_allowed={**PASS_ALLOWED, "BBB": 150.0, "DDD": 150.0})
    assert _group(result, "WR1") == {"BBB WR1"}
    assert _group(result, "WR2") == {"AAA WR2"}  # AAA plays BBB
    # WR3 goes by how DDD treats each WR role; with no history, the WR1.
    assert _group(result, "WR3") == {"CCC WR1"}


def test_wr3_takes_the_role_the_defense_handles_worse():
    vs = {"wr_role": {"DDD": {"WR1": 3.0, "WR2": -2.0}}}
    result, notes = _run(defense_vs=vs)
    assert _group(result, "WR3") == {"CCC WR1"}
    vs = {"wr_role": {"DDD": {"WR1": -3.0, "WR2": 2.0}}}
    result, notes = _run(defense_vs=vs)
    assert _group(result, "WR3") == {"CCC WR2"}
    assert any("DDD allows +2.0 FD pts/game to WR2s" in n and "(-3.0 to WR1s)" in n for n in notes)


def test_wr3_passes_over_a_team_starting_a_backup_qb():
    pool, stats = _pool_and_stats()
    pool = pool[pool["name"] != "CCC QB"]
    pool = pd.concat([pool, pd.DataFrame([_player("CCC Backup", "QB", "CCC", 6000, 3)])])
    result, notes = _run(pool, stats)
    assert any("WR3 passes over CCC" in n for n in notes)
    # FFF is the next-highest implied team (24.5) in games 2 and 3.
    assert _group(result, "WR3") == {"FFF WR1"}


def test_wr3_needs_an_18_percent_target_share():
    pool, stats = _pool_and_stats()
    stats.loc[stats["team"] == "CCC", "target_share"] = 0.10
    result, _ = _run(pool, stats)
    assert _group(result, "WR3") == {"FFF WR1"}


def test_te_needs_goal_line_targets_on_a_23_plus_team():
    pool, stats = _pool_and_stats()
    stats = pd.concat([stats, pd.DataFrame([
        _stat("EEE TE", "TE", "EEE", share=0.15, goal_line=3),   # implied 23.5
        _stat("AAA TE", "TE", "AAA", share=0.10, goal_line=1),
        _stat("DDD TE", "TE", "DDD", share=0.30, goal_line=5),   # implied 20.0, too low
        _stat("CCC TE", "TE", "CCC", share=0.25, goal_line=0),   # none inside the 10
    ])])
    result, notes = _run(pool, stats)
    assert _group(result, "TE") == {"EEE TE"}
    assert any("3 targets inside the 10" in n for n in notes)


def test_te_ties_go_to_the_softer_te_defense():
    pool, stats = _pool_and_stats()
    stats = pd.concat([stats, pd.DataFrame([
        _stat("AAA TE", "TE", "AAA", share=0.15, goal_line=2),   # faces BBB
        _stat("FFF TE", "TE", "FFF", share=0.15, goal_line=2),   # faces EEE
    ])])
    result, notes = _run(pool, stats, defense_vs={"te": {"BBB": 6.0, "EEE": 12.0}})
    assert _group(result, "TE") == {"FFF TE"}
    assert any("EEE allows 12.0 FD pts/game to TEs" in n for n in notes)


def test_te_rule_relaxes_without_goal_line_targets():
    result, notes = _run()
    assert len(_group(result, "TE")) == 1
    assert any("TE rule relaxed" in n for n in notes)


def test_def_faces_none_of_the_picks_and_flex_cant_face_it():
    result, _ = _run()
    picked = result[result["required_group"].notna() & (result["position"] != "DEF")]
    def_row = result[result["required_group"] == "DEF"].iloc[0]
    assert def_row["opponent"] not in set(picked["team"])
    facing = result[(result["team"] == def_row["opponent"]) & (result["position"] != "DEF")]
    assert facing.empty


def test_lineup_has_at_most_four_players_from_one_game():
    pool, stats = _pool_and_stats()
    pool.loc[pool["team"].isin(["AAA", "BBB"]), "projected_points"] = 40  # the optimizer wants them all
    result, _ = _run(pool, stats)
    lineup = optimize_lineup(60000, result)
    assert len(lineup[lineup["team"].isin(["AAA", "BBB"])]) == 4


def test_flex_is_a_wr2_rb2_or_te_from_the_second_or_third_game():
    result, notes = _run()
    flex = result[result["required_group"] == "FLEX"]
    allowed = {f"{t} {role}" for t in ("CCC", "DDD", "EEE", "FFF") for role in ("WR2", "RB2", "TE")}
    assert len(flex) > 0
    assert set(flex["name"]) <= allowed
    assert any(n.startswith("FLEX, one of") and "from the second and third games" in n for n in notes)
    lineup = optimize_lineup(60000, result)
    assert len(set(lineup["name"]) & set(flex["name"])) == 1


def test_flex_leaves_out_players_facing_the_def_and_over_budget():
    pool, stats = _pool_and_stats()
    pool.loc[pool["name"] == "EEE RB2", "salary"] = 30000
    result, _ = _run(pool, stats)
    flex = set(result.loc[result["required_group"] == "FLEX", "name"])
    def_opp = result.loc[result["required_group"] == "DEF", "opponent"].iloc[0]  # DDD
    assert not any(name.startswith(def_opp) for name in flex)
    assert "EEE RB2" not in flex


def test_rule_picks_leave_room_under_the_cap():
    pool, stats = _pool_and_stats()
    pool.loc[pool["name"] == "FFF RB1", "salary"] = 20000  # top-ranked RB, unaffordable
    result, notes = _run(pool, stats)
    assert "FFF RB1" not in _group(result, "RB1") | _group(result, "RB2")
    assert any("FFF RB1 would leave too little of the $60,000 cap" in n for n in notes)
    lineup = optimize_lineup(60000, result)
    assert lineup["salary"].sum() <= 60000


def test_defense_vs_receivers_from_weekly_stats():
    rows = []
    # XXX's WR1 (A) and WR2 (B) average 10 and 6 elsewhere; vs DDD they score 16 and 2.
    for week, opp, a, b in ((1, "P", 10, 6), (2, "Q", 10, 6), (3, "DDD", 16, 2)):
        rows += [
            {"season_type": "REG", "week": week, "team": "XXX", "opponent_team": opp, "position": "WR",
             "player_id": "A", "target_share": 0.3, "fantasy_points": a, "receptions": 0},
            {"season_type": "REG", "week": week, "team": "XXX", "opponent_team": opp, "position": "WR",
             "player_id": "B", "target_share": 0.2, "fantasy_points": b, "receptions": 0},
            {"season_type": "REG", "week": week, "team": "XXX", "opponent_team": opp, "position": "TE",
             "player_id": "T", "target_share": 0.1, "fantasy_points": 4, "receptions": 4},
        ]
    vs = defense_vs_receivers(pd.DataFrame(rows))
    assert vs["wr_role"]["DDD"] == {"WR1": 6.0, "WR2": -4.0}
    assert vs["te"]["DDD"] == 6.0  # 4 + 0.5 * 4 receptions


def test_def_faces_the_lowest_implied_offense():
    result, notes = _run()
    # DDD is a 9-point home underdog in the 49-point game: implied 20.0, the
    # lowest in the top-3 games (GGG's 17.0 is in the fourth).
    assert _group(result, "DEF") == {"CCC Defense"}
    assert any("implied 20.0" in n for n in notes)


def test_def_skips_an_offense_already_stacked_in_the_lineup():
    odds = ODDS.copy()
    odds.loc[odds["home"] == "BBB", "home_spread"] = -30.0  # AAA implied 11.0, lowest on the slate
    result, notes = _run(odds=odds)
    assert _group(result, "WR2") == {"AAA WR1"}
    assert _group(result, "DEF") == {"CCC Defense"}
    assert any("not already in the lineup (AAA is)" in n for n in notes)


def test_wr3_skipped_with_only_one_afternoon_game():
    result, notes = _run(odds=ODDS[ODDS["home"].isin(["JJJ", "BBB"])])
    assert _group(result, "WR3") == set()
    assert any("WR3 rule skipped" in n for n in notes)


def test_optimizer_builds_a_lineup_from_the_alpha_picks():
    result, _ = _run()
    lineup = optimize_lineup(60000, result)
    locks = result["required_group"].notna() & (result["required_group"] != "FLEX")
    assert set(result.loc[locks, "name"]) <= set(lineup["name"])


def test_alpha_strategy_falls_back_to_matchup_filters(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("odds unavailable")

    monkeypatch.setattr(player_pool, "build_alpha_pool", _boom)
    monkeypatch.setattr(player_pool, "build_eligibility", _boom)
    pool, _ = _pool_and_stats()
    result, note = player_pool._apply_alpha_strategy(pool)
    assert "Alpha Rule unavailable (odds unavailable)" in note
    assert len(result) == len(pool)
