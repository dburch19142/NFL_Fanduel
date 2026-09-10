from flask import Flask, render_template, request, redirect, url_for, flash

import pandas as pd

from optimizer import optimize_lineup, InfeasibleLineupError, DEFAULT_SALARY_CAP
from fanduel_import import FanDuelImportError
from player_pool import build_player_pool, save_uploaded_csv

app = Flask(__name__)
app.secret_key = "dev-only-not-for-production"

TEAMS_CSV = "data/teams.csv"


@app.route("/")
def dashboard():
    teams = pd.read_csv(TEAMS_CSV)
    teams["win_pct"] = (teams["wins"] / (teams["wins"] + teams["losses"] + teams["ties"])).round(3)
    teams["point_diff"] = teams["points_for"] - teams["points_against"]
    teams = teams.sort_values(by="win_pct", ascending=False).reset_index(drop=True)
    return render_template("dashboard.html", teams=teams.to_dict(orient="records"))


@app.route("/optimizer", methods=["GET", "POST"])
def optimizer_page():
    lineup = None
    error = None
    salary_cap = DEFAULT_SALARY_CAP

    players, pool_source = build_player_pool()

    if request.method == "POST":
        try:
            salary_cap = float(request.form.get("salary_cap", DEFAULT_SALARY_CAP))
            if salary_cap <= 0:
                raise ValueError("Salary cap must be positive.")
            result = optimize_lineup(salary_cap, players)
            lineup = {
                "players": result.to_dict(orient="records"),
                "total_salary": int(result["salary"].sum()),
                "total_points": round(float(result["projected_points"].sum()), 1),
            }
        except InfeasibleLineupError as exc:
            error = str(exc)
        except ValueError:
            error = "Please enter a valid positive number for the salary cap."

    return render_template(
        "optimizer.html",
        lineup=lineup,
        error=error,
        salary_cap=salary_cap,
        pool_source=pool_source,
        player_count=len(players),
    )


@app.route("/upload", methods=["POST"])
def upload_fanduel_csv():
    file = request.files.get("fanduel_csv")
    if not file or file.filename == "":
        flash("Please choose a FanDuel export CSV file first.", "error")
        return redirect(url_for("optimizer_page"))

    try:
        parsed = save_uploaded_csv(file)
        flash(f"Loaded {len(parsed)} players from {file.filename}.", "success")
    except FanDuelImportError as exc:
        flash(str(exc), "error")

    return redirect(url_for("optimizer_page"))


if __name__ == "__main__":
    app.run(debug=True)
