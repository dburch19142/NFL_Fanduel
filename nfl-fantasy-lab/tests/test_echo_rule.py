"""Unit tests for the Echo Rule strategy (no network needed). The slate and
stats come from the Alpha Rule tests, plus season carries for each team's
backs: RB1 has 150, RB2 40 (a 21% committee share)."""
import pandas as pd

import player_pool
from echo_rule import apply_echo_rule, echo_games
from optimizer import optimize_lineup
from test_alpha_rule import ODDS, PASS_ALLOWED, RUSH_ALLOWED, TEAMS, _group, _pool_and_stats, _stat


def _stats_with_carries(extra=()):
    _, stats = _pool_and_stats()
    rbs = [_stat(f"{t} RB{n}", "RB", t, carries=c) for t in TEAMS for n, c in ((1, 150), (2, 40))]
    return pd.concat([stats, pd.DataFrame(rbs + list(extra))], ignore_index=True)


def _run(pool=None, stats=None, odds=ODDS):
    base_pool, _ = _pool_and_stats()
    return apply_echo_rule(
        base_pool if pool is None else pool, odds, _stats_with_carries() if stats is None else stats,
        RUSH_ALLOWED, PASS_ALLOWED,
    )


def _flex_names(notes):
    return next(n for n in notes if n.startswith("FLEX, "))


def test_games_under_46_are_left_out():
    odds = ODDS.assign(total=ODDS["total"].where(ODDS["home"] != "FFF", 45.0))
    used, note = echo_games(odds, set(TEAMS))
    assert "FFF" not in set(used["home"])
    assert "46+" in note
    pool, _ = _run(odds=odds)
    assert set(pool["team"]) == {"AAA", "BBB", "CCC", "DDD"}


def test_game_rule_relaxes_with_fewer_than_two_46_plus_games():
    odds = ODDS.assign(total=ODDS["total"].where(ODDS["home"] == "BBB", 44.0))
    used, note = echo_games(odds, set(TEAMS))
    assert used is odds
    assert "using the top 3 instead" in note


def test_committee_backs_are_dropped():
    pool, notes = _run()
    assert not (pool["name"] == "FFF RB2").any()
    assert any(n.startswith("Committee backs dropped") for n in notes)


def test_fill_in_for_an_injured_lead_back_stays_in_the_pool():
    base_pool, _ = _pool_and_stats()
    pool, _ = _run(pool=base_pool[base_pool["name"] != "FFF RB1"])
    assert "FFF RB2" in _group(pool, "RB1")


def test_alpha_flex_note_is_replaced():
    _, notes = _run()
    assert sum(n.startswith("FLEX") and "one of" in n for n in notes) == 1
    assert "pts/$1K" in _flex_names(notes)


def test_flex_needs_a_real_role():
    # The second and third games' TEs have no target share on record.
    _, notes = _run()
    role = next(n for n in notes if n.startswith("FLEX passes over (role)"))
    assert "EEE TE" in role and "EEE TE" not in _flex_names(notes)


def test_flex_pass_catchers_on_a_big_favorite_are_passed_over():
    # CCC is favored by 9 over DDD.
    _, notes = _run()
    script = next(n for n in notes if n.startswith("FLEX passes over (game script)"))
    assert "CCC WR2" in script


def test_flex_rbs_need_a_team_that_leads():
    # DDD (20.0) and EEE (23.5) are underdogs implied for under 24.
    result, notes = _run()
    flex_rbs = result[(result["required_group"] == "FLEX") & (result["position"] == "RB")]
    assert not set(flex_rbs["team"]) & {"DDD", "EEE"}
    if "EEE RB1" not in set(result.loc[result["required_group"].notna(), "name"]):
        assert any("EEE RB1" in n for n in notes if n.startswith("FLEX passes over (game script)"))


def test_flex_can_take_a_second_stack_piece():
    # CCC's TE takes the TE slot, leaving room in the top game for the QB's next WR.
    base_pool, _ = _pool_and_stats()
    pool = base_pool.copy()
    pool.loc[pool["name"] == "BBB WR2", "salary"] = 4000
    stats = _stats_with_carries([_stat("CCC TE", "TE", "CCC", share=0.2, goal_line=3)])
    result, notes = _run(pool=pool, stats=stats)
    assert "CCC TE" in _group(result, "TE")
    assert "BBB WR2" in _group(result, "FLEX")
    assert "BBB WR2 (BBB WR, 2.8 pts/$1K, stack, the QB's next WR)" in _flex_names(notes)


def test_value_floor_keeps_only_players_paying_off_their_salary():
    base_pool, _ = _pool_and_stats()
    pool = base_pool.copy()
    for name in ("EEE WR2", "FFF WR2"):
        pool.loc[pool["name"] == name, ["salary", "projected_points"]] = (4000, 12.0)
    result, notes = _run(pool=pool)
    assert _group(result, "FLEX") == {"EEE WR2", "FFF WR2"}
    assert not any("best values" in n for n in notes)


def test_value_floor_relaxes_to_the_best_values():
    _, notes = _run()
    assert any("keeping the" in n and "best values" in n for n in notes)


def test_cheap_rule_picks_below_value_are_flagged():
    _, notes = _run()
    assert any(n.startswith("Value check") for n in notes)


def test_optimizer_builds_a_lineup_from_the_echo_picks():
    pool, _ = _run()
    lineup = optimize_lineup(60000, pool)
    flex = _group(pool, "FLEX")
    assert flex & set(lineup["name"])


def test_echo_strategy_falls_back_to_matchup_filters(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("odds unavailable")

    monkeypatch.setattr(player_pool, "build_echo_pool", _boom)
    monkeypatch.setattr(player_pool, "build_eligibility", _boom)
    pool, _ = _pool_and_stats()
    result, note = player_pool._apply_echo_strategy(pool)
    assert "Echo Rule unavailable (odds unavailable)" in note
    assert len(result) == len(pool)


def test_cheap_rule_pick_below_value_gives_way_to_a_better_value():
    # The bring-back (AAA WR1) costs $5,900 for 10 pts (1.7 per $1K); BBB WR2,
    # in the same game with a 20% target share, is $4,000 for 11 (2.8).
    base_pool, _ = _pool_and_stats()
    pool = base_pool.copy()
    pool.loc[pool["name"] == "AAA WR1", ["salary", "projected_points"]] = (5900, 10.0)
    pool.loc[pool["name"] == "BBB WR2", "salary"] = 4000
    result, notes = _run(pool=pool)
    assert _group(result, "WR2") == {"BBB WR2"}
    assert any("WR2 AAA WR1 (1.7) -> BBB WR2 (BBB, $4,000, 2.8)" in n for n in notes)


def test_cheap_rule_pick_is_kept_when_no_better_value_has_a_role():
    # No TE on the slate has a target share on record.
    result, notes = _run()
    kept = next(n for n in notes if n.startswith("Value check, no better value"))
    assert "AAA TE" in kept and "AAA TE" in _group(result, "TE")


def _downgrade_pool(wr3_points):
    # The bring-back (AAA WR1, $5,900, 12 pts) is under the value line, and
    # the only better value, BBB WR2 ($9,000, 24 pts), doesn't fit until the
    # WR3 (CCC WR1, now $13,000) moves down.
    base_pool, _ = _pool_and_stats()
    pool = base_pool.copy()
    pool.loc[pool["name"] == "AAA WR1", ["salary", "projected_points"]] = (5900, 12.0)
    pool.loc[pool["name"] == "BBB WR2", ["salary", "projected_points"]] = (9000, 24.0)
    pool.loc[pool["name"] == "CCC WR1", ["salary", "projected_points"]] = (13000, wr3_points)
    return pool


def test_most_expensive_pick_is_downgraded_to_make_room():
    result, notes = _run(pool=_downgrade_pool(15.0))
    assert _group(result, "WR2") == {"BBB WR2"}
    assert len(_group(result, "WR3") & {"EEE WR1", "FFF WR1"}) == 1
    assert any(n.startswith("Value check, most expensive pick downgraded") and "WR3 CCC WR1" in n
               and "freeing $6,000 for AAA WR1" in n for n in notes)


def test_no_downgrade_when_it_costs_projected_points():
    # Moving a 30-point WR3 down to a 14-point WR to gain 12 points elsewhere loses 4.
    result, notes = _run(pool=_downgrade_pool(30.0))
    assert "CCC WR1" in _group(result, "WR3")
    assert any("no better value" in n and "AAA WR1" in n for n in notes)


def test_swaps_leave_room_for_a_flex_that_passes():
    # Swapping the bring-back (AAA WR1, $5,900) straight to BBB WR2 ($14,000)
    # would leave $5,000, under every passing FLEX candidate ($6,000), so a
    # pick is downgraded too and FLEX is still the Echo Rule's.
    base_pool, _ = _pool_and_stats()
    pool = base_pool.copy()
    pool.loc[pool["name"] == "AAA WR1", ["salary", "projected_points"]] = (5900, 12.0)
    pool.loc[pool["name"] == "BBB WR2", ["salary", "projected_points"]] = (14000, 40.0)
    result, notes = _run(pool=pool)
    assert _group(result, "WR2") == {"BBB WR2"}
    assert any(n.startswith("Value check, most expensive pick downgraded") for n in notes)
    flex = result[result["required_group"] == "FLEX"]
    locked = result.loc[result["required_group"].notna() & (result["required_group"] != "FLEX"), "salary"].sum()
    assert not flex.empty and (flex["salary"] <= 60000 - locked).all()
    assert not any("FLEX is the optimizer's choice" in n for n in notes)


def _rushing(offense, defense):
    # offense/defense: {team: rank}; everyone else ranks 20th.
    return pd.DataFrame([{"team": t, "offense_rank": offense.get(t, 20), "defense_rank": defense.get(t, 20),
                          "rush_ypg": 100.0, "rush_allowed_ypg": 100.0} for t in TEAMS]).set_index("team")


def _run_beta(rushing, pool=None):
    base_pool, _ = _pool_and_stats()
    return apply_echo_rule(base_pool if pool is None else pool, ODDS, _stats_with_carries(),
                           RUSH_ALLOWED, PASS_ALLOWED, rushing_ranks=rushing)


def test_beta_backs_take_both_rb_slots_even_outside_the_alpha_games():
    # HHH (No. 1 rush O) faces GGG (No. 1 worst rush D) in the fourth Sunday
    # afternoon game, which the Alpha Rule leaves out; EEE faces FFF (No. 2).
    result, notes = _run_beta(_rushing({"HHH": 1, "EEE": 2}, {"GGG": 1, "FFF": 2}))
    assert _group(result, "RB1") == {"HHH RB1"} and _group(result, "RB2") == {"EEE RB1"}
    assert set(result.loc[result["position"] == "RB", "team"]) <= {"HHH", "EEE"}
    assert any("every other RB is dropped" in n for n in notes)
    lineup = optimize_lineup(60000, result)
    assert set(lineup.loc[lineup["position"] == "RB", "team"]) <= {"HHH", "EEE"}


def test_one_beta_back_takes_rb2_and_the_alpha_rb1_stays():
    alpha_result, _ = _run()
    alpha_rb1 = _group(alpha_result, "RB1")
    result, notes = _run_beta(_rushing({"HHH": 1}, {"GGG": 1}))
    assert _group(result, "RB2") == {"HHH RB1"}
    assert _group(result, "RB1") == alpha_rb1
    assert any("he takes RB2" in n for n in notes)


def test_no_beta_team_keeps_the_alpha_rbs():
    alpha_result, _ = _run()
    result, notes = _run_beta(_rushing({}, {}))
    assert _group(result, "RB1") == _group(alpha_result, "RB1")
    assert any(n.startswith("Beta RBs: no team") for n in notes)


def test_beta_back_facing_the_def_is_skipped():
    # The DEF faces DDD, so DDD's backs never come back into the pool.
    result, _ = _run_beta(_rushing({"DDD": 1, "HHH": 2}, {"CCC": 1, "GGG": 2}))
    assert not ((result["team"] == "DDD") & (result["position"] == "RB")).any()
    assert _group(result, "RB2") == {"HHH RB1"}


def test_echo_keeps_the_alpha_two_player_bring_back():
    # BBB's WR1 at a 20% target share: no stack, AAA's WR1 and TE instead.
    stats = _stats_with_carries()
    stats.loc[stats["match_key"] == "bbb wr1", "target_share"] = 0.20
    result, notes = _run(stats=stats)
    assert not _group(result, "WR1")
    assert _group(result, "WR2") == {"AAA WR1"} and _group(result, "TE") == {"AAA TE"}
    assert any(n.startswith("Two-player bring-back") for n in notes)
    assert optimize_lineup(60000, result)["name"].isin(["BBB QB"]).any()
