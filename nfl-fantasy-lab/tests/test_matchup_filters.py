"""Unit tests for the pure-logic helpers in matchup_filters.py.

build_eligibility() itself pulls three live datasets from nflverse and
isn't unit-tested directly here (matching how real_stats.py's equivalent
live-data function isn't either) -- these tests cover the ranking/fairness
logic that function relies on, using data we control.
"""
import pandas as pd

from matchup_filters import _per_game, _top_n_keys


def test_per_game_divides_by_games_played_not_just_summed():
    df = pd.DataFrame([
        {"player_display_name": "Player A", "team": "AAA", "week": 1, "passing_yards": 300},
        {"player_display_name": "Player A", "team": "AAA", "week": 2, "passing_yards": 100},
        {"player_display_name": "Player B", "team": "BBB", "week": 1, "passing_yards": 150},
    ])
    result = _per_game(df, ["player_display_name", "team"], ["passing_yards"])
    a = result[result["player_display_name"] == "Player A"].iloc[0]
    b = result[result["player_display_name"] == "Player B"].iloc[0]
    assert a["games"] == 2
    assert a["passing_yards"] == 200  # (300+100)/2, not the raw 400 total
    assert b["passing_yards"] == 150


def test_top_n_keys_includes_everyone_tied_at_the_cutoff():
    # Six players tied for what would be "10th place" if only the top 5
    # distinct values counted -- a plain nlargest(5) would arbitrarily keep
    # only some of the tied group; fair ranking keeps all of them.
    df = pd.DataFrame([
        {"player_display_name": f"Player {i}", "team": "AAA", "rushing_yards": 100}
        for i in range(6)
    ] + [
        {"player_display_name": "Best Player", "team": "AAA", "rushing_yards": 200},
    ])
    keys = _top_n_keys(df, "rushing_yards", 5)
    # "Best Player" (rank 1) plus all 6 tied at rank 2 = 7 total, even
    # though n=5.
    assert len(keys) == 7


def test_top_n_keys_normalizes_player_names():
    df = pd.DataFrame([
        {"player_display_name": "A.J. Brown", "team": "PHI", "receiving_yards": 100},
    ])
    keys = _top_n_keys(df, "receiving_yards", 10)
    assert ("aj brown", "PHI") in keys
