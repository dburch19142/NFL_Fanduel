import os

from flask import Flask, render_template, request, redirect, url_for, flash

import pandas as pd

# All data paths in this app (data/teams.csv, data/players.csv, data/cache, ...)
# are relative, so make sure they resolve against this file's directory
# regardless of what directory the process was launched from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from optimizer import (
    optimize_lineups,
    InfeasibleLineupError,
    PlayerNotFoundError,
    DEFAULT_SALARY_CAP,
)
from fanduel_import import FanDuelImportError
from player_pool import build_player_pool, save_uploaded_csv, STRATEGY_MATCHUP, STRATEGY_VEGAS
from game_predictor import (
    build_training_frame,
    current_season_form,
    evaluate_model,
    load_game_results,
    predict_matchup,
    split_by_season,
    train_model,
)
from real_stats import latest_available_season

LINEUP_COUNT = 1

app = Flask(__name__)
app.secret_key = "dev-only-not-for-production"

TEAMS_CSV = "data/teams.csv"
PREDICTOR_SEASONS_OF_HISTORY = 5

DIVISIONS = {
    "AFC East": ["Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets"],
    "AFC North": ["Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers"],
    "AFC South": ["Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans"],
    "AFC West": ["Denver Broncos", "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers"],
    "NFC East": ["Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington Commanders"],
    "NFC North": ["Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings"],
    "NFC South": ["Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers"],
    "NFC West": ["Arizona Cardinals", "Los Angeles Rams", "San Francisco 49ers", "Seattle Seahawks"],
}


@app.route("/")
def dashboard():
    teams = pd.read_csv(TEAMS_CSV)
    teams["win_pct"] = (teams["wins"] / (teams["wins"] + teams["losses"] + teams["ties"])).round(3)
    teams["point_diff"] = teams["points_for"] - teams["points_against"]
    teams = teams.sort_values(by=["win_pct", "point_diff"], ascending=False)
    divisions = [
        {"name": name, "teams": teams[teams["team"].isin(members)].to_dict(orient="records")}
        for name, members in DIVISIONS.items()
    ]
    return render_template("dashboard.html", divisions=divisions)


@app.route("/optimizer", methods=["GET", "POST"])
def optimizer_page():
    lineups = None
    error = None
    salary_cap = DEFAULT_SALARY_CAP
    must_include = ""
    strategy = request.values.get("strategy", STRATEGY_MATCHUP)
    if strategy not in (STRATEGY_MATCHUP, STRATEGY_VEGAS):
        strategy = STRATEGY_MATCHUP

    players, pool_source = build_player_pool(strategy=strategy)
    strategy_notes = players.attrs.get("strategy_notes", [])

    if request.method == "POST":
        must_include = request.form.get("must_include", "").strip()
        try:
            salary_cap = float(request.form.get("salary_cap", DEFAULT_SALARY_CAP))
            if salary_cap <= 0:
                raise ValueError("Salary cap must be positive.")
            results = optimize_lineups(
                salary_cap, players, count=LINEUP_COUNT,
                required_name=must_include or None,
            )
            lineups = [
                {
                    "players": result.to_dict(orient="records"),
                    "total_salary": int(result["salary"].sum()),
                    "total_points": round(float(result["projected_points"].sum()), 1),
                }
                for result in results
            ]
        except InfeasibleLineupError as exc:
            error = str(exc)
        except PlayerNotFoundError as exc:
            error = str(exc)
        except ValueError:
            error = "Please enter a valid positive number for the salary cap."

    return render_template(
        "optimizer.html",
        lineups=lineups,
        error=error,
        salary_cap=salary_cap,
        pool_source=pool_source,
        player_count=len(players),
        lineup_count=LINEUP_COUNT,
        must_include=must_include,
        strategy=strategy,
        strategy_notes=strategy_notes,
    )


def _train_game_predictor():
    """Trains the game-outcome model on all but the most recent season and
    evaluates it on that season, so the reported metrics reflect performance
    on games the model never saw during training."""
    latest_season = latest_available_season()
    seasons = list(range(latest_season - PREDICTOR_SEASONS_OF_HISTORY + 1, latest_season + 1))

    games = load_game_results(seasons)
    frame = build_training_frame(games)
    train, test = split_by_season(frame, test_season=latest_season)

    model = train_model(train)
    metrics = evaluate_model(model, test)
    form = current_season_form(games, season=latest_season).sort_values("team")

    return model, metrics, form, latest_season


@app.route("/predictor", methods=["GET", "POST"])
def predictor_page():
    model, metrics, form, season = _train_game_predictor()
    team_form = form.set_index("team").to_dict(orient="index")
    teams = sorted(team_form.keys())

    prediction = None
    error = None
    home_team = request.form.get("home_team")
    away_team = request.form.get("away_team")

    if request.method == "POST":
        if not home_team or not away_team:
            error = "Please choose both a home and an away team."
        elif home_team == away_team:
            error = "Home and away teams must be different."
        else:
            probability = predict_matchup(model, form.set_index("team").loc[home_team], form.set_index("team").loc[away_team])
            prediction = {
                "home_team": home_team,
                "away_team": away_team,
                "home_win_probability": round(probability * 100, 1),
                "away_win_probability": round((1 - probability) * 100, 1),
            }

    return render_template(
        "predictor.html",
        metrics=metrics,
        season=season,
        teams=teams,
        team_form=team_form,
        prediction=prediction,
        error=error,
        home_team=home_team,
        away_team=away_team,
    )


@app.route("/upload", methods=["POST"])
def upload_fanduel_csv():
    file = request.files.get("fanduel_csv")
    if not file or file.filename == "":
        flash("Please choose a FanDuel export CSV file first.", "error")
        return redirect(url_for("optimizer_page"))

    try:
        parsed = save_uploaded_csv(file)
        excluded = parsed.attrs.get("excluded_injured_count", 0)
        message = f"Loaded {len(parsed)} players from {file.filename}."
        if excluded:
            message += f" Omitted {excluded} ruled out, doubtful, or otherwise unavailable."
        flash(message, "success")
    except FanDuelImportError as exc:
        flash(str(exc), "error")

    return redirect(url_for("optimizer_page"))


if __name__ == "__main__":
    # use_reloader=False: the reloader re-execs the process using the original
    # (possibly relative) launch command, which breaks once the chdir() above
    # has already moved the process into this file's directory.
    app.run(debug=True, use_reloader=False, port=int(os.environ.get("PORT", 5000)))
