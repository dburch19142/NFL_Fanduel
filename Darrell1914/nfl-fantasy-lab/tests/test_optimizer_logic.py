"""Unit tests for the lineup optimizer (no browser needed).

Roster shape matches FanDuel's NFL classic contest: 1 QB, 2 RB, 3 WR, 1 TE,
1 FLEX, 1 DEF, $60,000 default cap. The minimum possible 9-man lineup with
the bundled sample data costs about $47,200, so test caps below that are
expected to be infeasible.
"""
import pandas as pd
import pytest

from optimizer import (
    optimize_lineup,
    InfeasibleLineupError,
    load_players,
    ROSTER_SLOTS,
    DEFAULT_SALARY_CAP,
    _dedupe_players,
)

PLAYERS = load_players("data/players.csv")


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
