"""Unit tests for the pure-logic helpers in matchup_filters.py.

build_eligibility() itself pulls three live datasets from nflverse and
isn't unit-tested directly here (matching how real_stats.py's equivalent
live-data function isn't either) -- these tests cover the ranking/fairness
logic that function relies on, using data we control.
"""
import pandas as pd

from matchup_filters import _bottom_n_teams, _per_game


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


def test_bottom_n_teams_includes_everyone_tied_at_the_cutoff():
    # Six teams tied for 2nd-worst -- a plain nlargest(2) would arbitrarily
    # keep only some of that tied group; fair ranking keeps all of them.
    df = pd.DataFrame([
        {"team": f"Team {i}", "rush_yds_allowed": 150} for i in range(6)
    ] + [
        {"team": "Worst Team", "rush_yds_allowed": 200},
    ])
    worst = _bottom_n_teams(df, "rush_yds_allowed", 2)
    # "Worst Team" (rank 1) plus all 6 tied at rank 2 = 7 total, even
    # though n=2.
    assert len(worst) == 7
