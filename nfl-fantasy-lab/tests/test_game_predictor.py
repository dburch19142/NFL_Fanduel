"""Unit tests for the game outcome predictor (no browser or network needed)."""
import pandas as pd
import pytest

from game_predictor import (
    FEATURE_COLUMNS,
    _entering_game_stats,
    _team_game_log,
    build_training_frame,
    current_season_form,
    evaluate_model,
    predict_matchup,
    split_by_season,
    train_model,
)

# A small hand-built schedule: AAA is the league's best team (wins every game
# it plays), CCC is the worst (loses every game), and BBB/DDD split evenly.
# This keeps the win/loss pattern obvious enough to assert on directly.
GAMES = pd.DataFrame(
    [
        {"game_id": "2023_01", "season": 2023, "week": 1, "home_team": "AAA", "away_team": "BBB", "home_score": 30, "away_score": 10},
        {"game_id": "2023_02", "season": 2023, "week": 2, "home_team": "CCC", "away_team": "AAA", "home_score": 7, "away_score": 28},
        {"game_id": "2023_03", "season": 2023, "week": 3, "home_team": "AAA", "away_team": "DDD", "home_score": 24, "away_score": 20},
        {"game_id": "2023_04", "season": 2023, "week": 1, "home_team": "DDD", "away_team": "CCC", "home_score": 17, "away_score": 14},
        {"game_id": "2023_05", "season": 2023, "week": 2, "home_team": "BBB", "away_team": "CCC", "home_score": 21, "away_score": 6},
        {"game_id": "2024_01", "season": 2024, "week": 1, "home_team": "AAA", "away_team": "CCC", "home_score": 35, "away_score": 3},
        {"game_id": "2024_02", "season": 2024, "week": 2, "home_team": "BBB", "away_team": "AAA", "home_score": 13, "away_score": 27},
    ]
)


def test_team_game_log_has_two_rows_per_game():
    log = _team_game_log(GAMES)
    assert len(log) == len(GAMES) * 2
    assert set(log["team"]) == {"AAA", "BBB", "CCC", "DDD"}


def test_team_game_log_marks_winner():
    log = _team_game_log(GAMES)
    game_1 = log[log["game_id"] == "2023_01"]
    assert game_1.set_index("team").loc["AAA", "win"] == 1
    assert game_1.set_index("team").loc["BBB", "win"] == 0


def test_entering_stats_default_to_neutral_before_first_game():
    stats = _entering_game_stats(_team_game_log(GAMES))
    first_game = stats[(stats["team"] == "AAA") & (stats["season"] == 2023) & (stats["week"] == 1)].iloc[0]
    assert first_game["win_pct_entering"] == 0.5
    assert first_game["point_diff_entering"] == 0.0


def test_entering_stats_never_leak_the_current_or_future_game():
    # AAA wins its week-1 and week-2 games (2023), so entering week 3 it
    # should show a 1.0 win pct -- and that must NOT yet include week 3.
    stats = _entering_game_stats(_team_game_log(GAMES))
    week_3 = stats[(stats["team"] == "AAA") & (stats["season"] == 2023) & (stats["week"] == 3)].iloc[0]
    assert week_3["win_pct_entering"] == 1.0
    assert week_3["games_played_entering"] == 2


def test_build_training_frame_labels_home_win_correctly():
    frame = build_training_frame(GAMES)
    labels = frame.set_index("game_id")["home_win"]
    assert labels["2023_01"] == 1  # AAA (home) beat BBB
    assert labels["2023_02"] == 0  # CCC (home) lost to AAA

    for col in FEATURE_COLUMNS:
        assert col in frame.columns
    assert not frame[FEATURE_COLUMNS].isna().any().any()


def test_split_by_season_partitions_train_and_test():
    frame = build_training_frame(GAMES)
    train, test = split_by_season(frame, test_season=2024)
    assert set(train["season"]) == {2023}
    assert set(test["season"]) == {2024}
    assert len(train) + len(test) == len(frame)


def test_current_season_form_reflects_all_games_played_so_far():
    form = current_season_form(GAMES, season=2023).set_index("team")
    assert form.loc["AAA", "games_played"] == 3
    assert form.loc["AAA", "win_pct"] == 1.0
    assert form.loc["CCC", "win_pct"] == 0.0


def test_model_trains_and_evaluates_end_to_end():
    frame = build_training_frame(GAMES)
    train, test = split_by_season(frame, test_season=2024)

    model = train_model(train)
    metrics = evaluate_model(model, test)

    assert metrics["n_games"] == len(test)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2


def test_predict_matchup_returns_a_probability():
    frame = build_training_frame(GAMES)
    model = train_model(frame)

    form = current_season_form(GAMES, season=2023).set_index("team")
    probability = predict_matchup(model, form.loc["AAA"], form.loc["CCC"])

    assert 0.0 <= probability <= 1.0
    # AAA (undefeated) at home against CCC (winless) should be heavily favored.
    assert probability > 0.5
