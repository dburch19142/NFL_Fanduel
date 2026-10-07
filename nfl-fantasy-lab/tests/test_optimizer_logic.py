"""Unit tests for the lineup optimizer (no browser needed).

Roster shape matches FanDuel's NFL classic contest: 1 QB, 2 RB, 3 WR, 1 TE,
1 FLEX, 1 DEF, $60,000 default cap. The minimum possible 9-man lineup with
the bundled sample data costs $37,000, so test caps below that are expected
to be infeasible.
"""
import pandas as pd
import pytest

from optimizer import (
    optimize_lineup,
    optimize_lineups,
    InfeasibleLineupError,
    PlayerNotFoundError,
    load_players,
    ROSTER_SLOTS,
    DEFAULT_SALARY_CAP,
    _dedupe_players,
)

PLAYERS = load_players("data/players.csv")


def _qb_with_unambiguous_stackmate() -> tuple[str, str]:
    """Finds a (name, team) QB already in PLAYERS with a same-team WR/TE and
    a name that appears only once in the pool.

    data/players.csv is refreshed from real, live FanDuel data (see
    player_pool.py), so any specific player hardcoded here would eventually
    fall off the current week's slate and break this test -- picking one
    dynamically keeps the test valid regardless of which real players are in
    this week's data.
    """
    name_counts = PLAYERS["name"].value_counts()
    for _, qb in PLAYERS[PLAYERS["position"] == "QB"].iterrows():
        if name_counts[qb["name"]] != 1:
            continue
        teammates = PLAYERS[
            (PLAYERS["team"] == qb["team"]) & (PLAYERS["position"].isin(["WR", "TE"]))
        ]
        if not teammates.empty:
            return qb["name"], qb["team"]
    raise RuntimeError("No unambiguous QB with a same-team WR/TE found in data/players.csv.")


def test_default_salary_cap_matches_fanduel():
    assert DEFAULT_SALARY_CAP == 60000


def test_lineup_respects_salary_cap():
    lineup = optimize_lineup(DEFAULT_SALARY_CAP, PLAYERS)
    assert lineup["salary"].sum() <= DEFAULT_SALARY_CAP


def test_lineup_has_correct_roster_size():
    lineup = optimize_lineup(DEFAULT_SALARY_CAP, PLAYERS)
    total_slots = sum(ROSTER_SLOTS.values()) + 1  # +1 FLEX
    assert total_slots == 9
    assert len(lineup) == total_slots


def test_lineup_has_required_positions():
    lineup = optimize_lineup(DEFAULT_SALARY_CAP, PLAYERS)
    counts = lineup["position"].value_counts()
    assert counts["QB"] == 1
    assert counts["DEF"] == 1
    assert counts["RB"] >= 2
    assert counts["WR"] >= 3
    assert counts["TE"] >= 1


def test_higher_cap_never_scores_lower():
    cheap_lineup = optimize_lineup(48000, PLAYERS)
    rich_lineup = optimize_lineup(70000, PLAYERS)
    assert rich_lineup["projected_points"].sum() >= cheap_lineup["projected_points"].sum()


def test_infeasible_cap_raises():
    with pytest.raises(InfeasibleLineupError):
        optimize_lineup(500, PLAYERS)


def test_duplicate_player_row_is_never_picked_twice():
    # Duplicate the best-value RB (Christian McCaffrey) as a second row with
    # a different id but the same name/team, as would happen with a stray
    # duplicate CSV row or a showdown-slate MVP entry for the same person.
    mccaffrey = PLAYERS[PLAYERS["name"] == "Christian McCaffrey"].iloc[0]
    duplicate = mccaffrey.copy()
    players_with_dupe = pd.concat([PLAYERS, duplicate.to_frame().T], ignore_index=True)

    lineup = optimize_lineup(DEFAULT_SALARY_CAP, players_with_dupe)
    assert (lineup["name"] == "Christian McCaffrey").sum() <= 1


def test_dedupe_drops_exact_duplicate_rows():
    mccaffrey = PLAYERS[PLAYERS["name"] == "Christian McCaffrey"].iloc[0]
    duplicate = mccaffrey.copy()
    players_with_dupe = pd.concat([PLAYERS, duplicate.to_frame().T], ignore_index=True)

    deduped = _dedupe_players(players_with_dupe)
    assert (deduped["name"] == "Christian McCaffrey").sum() == 1
    assert len(deduped) == len(PLAYERS)


def test_lineup_stacks_qb_with_a_teammate():
    lineup = optimize_lineup(DEFAULT_SALARY_CAP, PLAYERS)
    qb_team = lineup.loc[lineup["position"] == "QB", "team"].iloc[0]
    catchers = lineup[lineup["position"].isin(["WR", "TE"])]
    assert qb_team in set(catchers["team"])


def test_qb_with_no_teammate_in_pool_is_never_picked():
    # A QB with no WR/TE from his team in the pool can never satisfy the
    # stack requirement, so the solver must skip him even though he's the
    # obvious pure points-per-dollar pick.
    lonely_qb = PLAYERS[PLAYERS["position"] == "QB"].iloc[0].copy()
    lonely_qb["name"] = "Lonely QB"
    lonely_qb["team"] = "ZZZ"
    lonely_qb["salary"] = 5000
    lonely_qb["projected_points"] = 999.0
    players_with_lonely_qb = pd.concat(
        [PLAYERS, lonely_qb.to_frame().T], ignore_index=True
    )

    lineup = optimize_lineup(DEFAULT_SALARY_CAP, players_with_lonely_qb)
    assert "Lonely QB" not in set(lineup["name"])


def test_optimize_lineups_returns_five_distinct_lineups():
    lineups = optimize_lineups(DEFAULT_SALARY_CAP, PLAYERS, count=5)
    assert len(lineups) == 5

    seen = set()
    for lineup in lineups:
        assert len(lineup) == 9
        assert lineup["salary"].sum() <= DEFAULT_SALARY_CAP
        key = frozenset(zip(lineup["name"], lineup["team"]))
        assert key not in seen, "lineup repeated exactly"
        seen.add(key)


def test_optimize_lineups_ranked_best_first():
    lineups = optimize_lineups(DEFAULT_SALARY_CAP, PLAYERS, count=5)
    totals = [lineup["projected_points"].sum() for lineup in lineups]
    assert totals == sorted(totals, reverse=True)


def test_optimize_lineups_each_lineup_still_stacks_the_qb():
    lineups = optimize_lineups(DEFAULT_SALARY_CAP, PLAYERS, count=5)
    for lineup in lineups:
        qb_team = lineup.loc[lineup["position"] == "QB", "team"].iloc[0]
        catchers = lineup[lineup["position"].isin(["WR", "TE"])]
        assert qb_team in set(catchers["team"])


def test_optimize_lineups_returns_fewer_when_pool_too_thin():
    # Exactly enough players to fill every slot with zero spares anywhere
    # except one extra RB for FLEX, so only a single valid 9-man roster
    # exists -- optimize_lineups should return just that one lineup rather
    # than erroring out trying to find 5.
    thin_pool = pd.DataFrame(
        [
            {"name": "QB1", "position": "QB", "team": "AAA", "salary": 6000, "projected_points": 20},
            {"name": "RB1", "position": "RB", "team": "AAA", "salary": 5000, "projected_points": 12},
            {"name": "RB2", "position": "RB", "team": "BBB", "salary": 4800, "projected_points": 11},
            {"name": "RB3", "position": "RB", "team": "BBB", "salary": 4500, "projected_points": 10},
            {"name": "WR1", "position": "WR", "team": "AAA", "salary": 5200, "projected_points": 13},
            {"name": "WR2", "position": "WR", "team": "BBB", "salary": 4900, "projected_points": 12},
            {"name": "WR3", "position": "WR", "team": "BBB", "salary": 4200, "projected_points": 10},
            {"name": "TE1", "position": "TE", "team": "AAA", "salary": 3000, "projected_points": 7},
            {"name": "DEF1", "position": "DEF", "team": "AAA", "salary": 2000, "projected_points": 5},
        ]
    )

    lineups = optimize_lineups(DEFAULT_SALARY_CAP, thin_pool, count=5)
    assert len(lineups) == 1


def test_optimize_lineups_infeasible_cap_raises():
    with pytest.raises(InfeasibleLineupError):
        optimize_lineups(500, PLAYERS, count=5)


def test_position_shortfall_raises_specific_message_naming_the_position():
    # Zero RBs at all -- a shortfall that has nothing to do with the salary
    # cap (e.g. what matchup-eligibility filtering can produce), so the
    # error should name RB specifically rather than blaming the cap.
    no_rb_pool = PLAYERS[PLAYERS["position"] != "RB"]
    with pytest.raises(InfeasibleLineupError, match="RB"):
        optimize_lineups(DEFAULT_SALARY_CAP, no_rb_pool, count=5)


def test_required_player_is_forced_into_every_lineup():
    qb_name, qb_team = _qb_with_unambiguous_stackmate()
    lineups = optimize_lineups(DEFAULT_SALARY_CAP, PLAYERS, count=5, required_name=qb_name)
    assert len(lineups) == 5
    for lineup in lineups:
        assert qb_name in set(lineup["name"])
        # The stack requirement must still hold for the locked-in QB too.
        catchers = lineup[lineup["position"].isin(["WR", "TE"])]
        assert qb_team in set(catchers["team"])


def test_required_player_not_found_raises():
    with pytest.raises(PlayerNotFoundError):
        optimize_lineups(DEFAULT_SALARY_CAP, PLAYERS, required_name="Nobody Real")


def test_required_player_ambiguous_without_team_raises():
    # Deliberately unremarkable salary/points so neither twin is a lineup pick
    # on its own merit -- isolates the test to the required-player/team
    # disambiguation logic, regardless of what real players happen to be in
    # PLAYERS this week.
    twin_a = pd.Series(
        {"name": "Same Name Guy", "position": "WR", "team": "AAA", "salary": 4000, "projected_points": 1.0}
    )
    twin_b = pd.Series(
        {"name": "Same Name Guy", "position": "WR", "team": "BBB", "salary": 4000, "projected_points": 1.0}
    )
    players_with_twins = pd.concat(
        [PLAYERS, twin_a.to_frame().T, twin_b.to_frame().T], ignore_index=True
    )

    with pytest.raises(PlayerNotFoundError):
        optimize_lineups(DEFAULT_SALARY_CAP, players_with_twins, required_name="Same Name Guy")

    # Naming the team disambiguates and succeeds.
    lineups = optimize_lineups(
        DEFAULT_SALARY_CAP, players_with_twins, count=1,
        required_name="Same Name Guy", required_team="AAA",
    )
    matches = lineups[0][lineups[0]["name"] == "Same Name Guy"]
    assert len(matches) == 1
    assert matches.iloc[0]["team"] == "AAA"


def test_dedupe_keeps_same_name_different_team_distinct():
    # Two different real players can share a name (e.g. two NFL WRs named
    # Mike Williams) -- dedup must key on name+team, not name alone, or one
    # of two genuinely different people would vanish from consideration.
    row = PLAYERS[PLAYERS["position"] == "WR"].iloc[0].copy()
    twin_a, twin_b = row.copy(), row.copy()
    twin_a["name"] = twin_b["name"] = "Same Name Guy"
    twin_a["team"], twin_b["team"] = "AAA", "BBB"

    players_with_twins = pd.concat(
        [PLAYERS, twin_a.to_frame().T, twin_b.to_frame().T], ignore_index=True
    )
    deduped = _dedupe_players(players_with_twins)
    matches = deduped[deduped["name"] == "Same Name Guy"]
    assert len(matches) == 2
    assert set(matches["team"]) == {"AAA", "BBB"}


def _synthetic_pool_with_opponents() -> pd.DataFrame:
    """A small, fully synthetic, roster-fillable pool with an "opponent"
    column, for testing the DEF anti-correlation rule in isolation.

    The QB and his only viable stack partner (WR1) are both on team OPP,
    and are effectively forced picks (the sole QB in the pool; the sole
    WR/TE on his team, so the QB-stack rule requires WR1 specifically).
    DEF1's opponent is OPP and scores far higher than DEF2, whose opponent
    is unrelated -- so DEF1 would be the clear optimal pick on points alone,
    unless the anti-correlation rule (DEF1 plays the same team as the
    already-forced QB/WR1) correctly rules it out.
    """
    rows = [
        {"name": "QB1", "position": "QB", "team": "OPP", "opponent": "DEF1TEAM", "salary": 6000, "projected_points": 30},
        {"name": "WR1", "position": "WR", "team": "OPP", "opponent": "DEF1TEAM", "salary": 6000, "projected_points": 30},
        {"name": "WR2", "position": "WR", "team": "X", "opponent": "Q", "salary": 4000, "projected_points": 10},
        {"name": "WR3", "position": "WR", "team": "Y", "opponent": "Q", "salary": 4000, "projected_points": 10},
        {"name": "WR4", "position": "WR", "team": "Z", "opponent": "Q", "salary": 4000, "projected_points": 10},
        {"name": "RB1", "position": "RB", "team": "X", "opponent": "Q", "salary": 4000, "projected_points": 10},
        {"name": "RB2", "position": "RB", "team": "Y", "opponent": "Q", "salary": 4000, "projected_points": 10},
        {"name": "RB3", "position": "RB", "team": "Z", "opponent": "Q", "salary": 4000, "projected_points": 10},
        {"name": "TE1", "position": "TE", "team": "X", "opponent": "Q", "salary": 4000, "projected_points": 10},
        {"name": "DEF1", "position": "DEF", "team": "DEF1TEAM", "opponent": "OPP", "salary": 3000, "projected_points": 50},
        {"name": "DEF2", "position": "DEF", "team": "DEF2TEAM", "opponent": "Q", "salary": 3000, "projected_points": 5},
    ]
    return pd.DataFrame(rows)


def test_def_never_paired_with_a_qb_or_receiver_it_is_playing_against():
    lineup = optimize_lineup(DEFAULT_SALARY_CAP, _synthetic_pool_with_opponents())
    names = set(lineup["name"])
    # QB1/WR1 are effectively forced (sole QB; sole same-team stack partner),
    # so DEF1 -- their opponent -- must be excluded even though it scores far
    # higher than DEF2, proving the anti-correlation rule is what's binding.
    assert "QB1" in names
    assert "WR1" in names
    assert "DEF1" not in names
    assert "DEF2" in names


def test_def_correlation_rule_skipped_without_opponent_column():
    # PLAYERS (data/players.csv) has no "opponent" column -- the rule must
    # be a no-op rather than erroring when that data isn't available.
    lineup = optimize_lineup(DEFAULT_SALARY_CAP, PLAYERS)
    assert len(lineup) == sum(ROSTER_SLOTS.values()) + 1
