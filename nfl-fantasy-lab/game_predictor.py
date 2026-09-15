"""Predicts NFL game winners from historical scores using logistic regression.

Pulls real game-by-game results from the nflverse project (via nfl_data_py),
turns each team's results earlier in the same season into pregame features
(win percentage and average point differential -- computed with a shift so a
game never sees its own outcome or a future one), and fits a scikit-learn
logistic regression that estimates the home team's win probability.

Evaluation uses a time-based split (train on past seasons, test on the most
recent one) rather than a random split, since shuffling games across time
would let the model implicitly "see the future" of a season it's tested on.
"""
import os
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from real_stats import latest_available_season

CACHE_DIR = "data/cache"
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # 1 day

FEATURE_COLUMNS = ["home_win_pct", "away_win_pct", "home_point_diff", "away_point_diff"]

GAME_COLUMNS = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]


def _cache_path(seasons: list[int]) -> str:
    return f"{CACHE_DIR}/game_results_{min(seasons)}_{max(seasons)}.csv"


def load_game_results(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """Returns completed regular-season games for the given seasons.

    Columns: game_id, season, week, home_team, away_team, home_score, away_score.
    """
    cache_file = _cache_path(seasons)
    if not force_refresh and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < CACHE_MAX_AGE_SECONDS:
            return pd.read_csv(cache_file)

    import nfl_data_py as nfl

    schedules = nfl.import_schedules(seasons)
    games = schedules[schedules["game_type"] == "REG"]
    games = games.dropna(subset=["home_score", "away_score"])[GAME_COLUMNS].copy()
    games["season"] = games["season"].astype(int)
    games["week"] = games["week"].astype(int)

    os.makedirs(CACHE_DIR, exist_ok=True)
    games.to_csv(cache_file, index=False)
    return games


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Reshapes one row per game into two rows per game (one per team)."""
    home = games.rename(
        columns={"home_team": "team", "away_team": "opponent", "home_score": "team_score", "away_score": "opp_score"}
    )
    away = games.rename(
        columns={"away_team": "team", "home_team": "opponent", "away_score": "team_score", "home_score": "opp_score"}
    )
    cols = ["game_id", "season", "week", "team", "opponent", "team_score", "opp_score"]
    log = pd.concat([home[cols], away[cols]], ignore_index=True)
    log["win"] = (log["team_score"] > log["opp_score"]).astype(int)
    log["point_diff"] = log["team_score"] - log["opp_score"]
    return log.sort_values(["season", "team", "week"]).reset_index(drop=True)


def _entering_game_stats(log: pd.DataFrame) -> pd.DataFrame:
    """Adds each team's season-to-date win pct and point diff *before* each game.

    Uses a neutral 0.5 win rate / 0 point diff for a team's first game of a
    season, since it has no prior results yet to base a feature on.
    """
    log = log.copy()
    grouped = log.groupby(["season", "team"])
    games_played_entering = grouped.cumcount()
    prior_wins = grouped["win"].cumsum() - log["win"]
    prior_point_diff_sum = grouped["point_diff"].cumsum() - log["point_diff"]

    log["games_played_entering"] = games_played_entering
    safe_denominator = games_played_entering.replace(0, pd.NA)
    log["win_pct_entering"] = (prior_wins / safe_denominator).fillna(0.5)
    log["point_diff_entering"] = (prior_point_diff_sum / safe_denominator).fillna(0.0)
    return log


def build_training_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Returns `games` plus each side's pregame features and the home_win label."""
    log = _entering_game_stats(_team_game_log(games))
    stat_cols = ["game_id", "team", "win_pct_entering", "point_diff_entering"]

    home_stats = log.merge(games[["game_id", "home_team"]], on="game_id")
    home_stats = home_stats[home_stats["team"] == home_stats["home_team"]][stat_cols].rename(
        columns={"win_pct_entering": "home_win_pct", "point_diff_entering": "home_point_diff"}
    )
    away_stats = log.merge(games[["game_id", "away_team"]], on="game_id")
    away_stats = away_stats[away_stats["team"] == away_stats["away_team"]][stat_cols].rename(
        columns={"win_pct_entering": "away_win_pct", "point_diff_entering": "away_point_diff"}
    )

    frame = games.merge(home_stats.drop(columns="team"), on="game_id").merge(
        away_stats.drop(columns="team"), on="game_id"
    )
    frame["home_win"] = (frame["home_score"] > frame["away_score"]).astype(int)
    return frame


def current_season_form(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Each team's actual win pct / avg point diff across all of `season` so far.

    Meant for predicting a game that hasn't been played yet, so (unlike the
    pregame features above) it deliberately includes the team's most recent
    result rather than shifting it out.
    """
    log = _team_game_log(games)
    season_log = log[log["season"] == season]
    return (
        season_log.groupby("team")
        .agg(games_played=("win", "size"), win_pct=("win", "mean"), point_diff=("point_diff", "mean"))
        .reset_index()
    )


def split_by_season(frame: pd.DataFrame, test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split: every other season trains, `test_season` tests."""
    return frame[frame["season"] != test_season], frame[frame["season"] == test_season]


def train_model(frame: pd.DataFrame) -> Pipeline:
    model = Pipeline([("scale", StandardScaler()), ("logreg", LogisticRegression())])
    model.fit(frame[FEATURE_COLUMNS], frame["home_win"])
    return model


def evaluate_model(model: Pipeline, frame: pd.DataFrame) -> dict:
    """Returns accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix."""
    X, y = frame[FEATURE_COLUMNS], frame["home_win"]
    predicted = model.predict(X)
    win_probability = model.predict_proba(X)[:, 1]
    return {
        "n_games": len(y),
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, win_probability),
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
    }


def predict_matchup(model: Pipeline, home_form: pd.Series, away_form: pd.Series) -> float:
    """Returns the home team's predicted win probability given each team's current form."""
    X = pd.DataFrame(
        [
            {
                "home_win_pct": home_form["win_pct"],
                "away_win_pct": away_form["win_pct"],
                "home_point_diff": home_form["point_diff"],
                "away_point_diff": away_form["point_diff"],
            }
        ]
    )
    return float(model.predict_proba(X[FEATURE_COLUMNS])[0, 1])


if __name__ == "__main__":
    latest = latest_available_season()
    seasons = list(range(latest - 4, latest + 1))

    games = load_game_results(seasons)
    frame = build_training_frame(games)
    train, test = split_by_season(frame, test_season=latest)

    model = train_model(train)
    metrics = evaluate_model(model, test)

    print(f"Trained on {len(train)} games from seasons {sorted(set(train['season']))}")
    print(f"Tested on {len(test)} games from the {latest} season\n")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    weights = dict(zip(FEATURE_COLUMNS, model.named_steps["logreg"].coef_[0]))
    print("\nStandardized feature weights (larger magnitude = bigger influence on the prediction):")
    for name, weight in weights.items():
        print(f"  {name}: {weight:+.3f}")
