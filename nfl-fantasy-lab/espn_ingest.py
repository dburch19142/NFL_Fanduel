"""Loads current NFL game data from ESPN's public (unofficial) API into a
local SQLite database, and exports each table as a CSV for Power BI.

Run any time to refresh -- during a game it pulls the live box score:

    python espn_ingest.py                    # current week
    python espn_ingest.py --weeks 1-3        # a range of weeks, same season
    python espn_ingest.py --season 2025 --seasontype 2 --weeks 18

Tables (data/espn_nfl.db, also written as data/powerbi/<table>.csv):
  games         one row per game: teams, score, status, week
  team_stats    one row per team per game per stat (long format)
  player_stats  one row per player per game per stat (long format)

Rows are upserted, so re-running refreshes in-progress games in place.
ESPN's API is undocumented and can change without notice.
"""
import argparse
import csv
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
DB_PATH = "data/espn_nfl.db"
CSV_DIR = "data/powerbi"

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    event_id      TEXT PRIMARY KEY,
    season        INTEGER,
    season_type   INTEGER,
    week          INTEGER,
    game_date     TEXT,
    name          TEXT,
    status        TEXT,
    status_detail TEXT,
    period        INTEGER,
    clock         TEXT,
    venue         TEXT,
    home_team_id  TEXT,
    home_abbr     TEXT,
    home_score    INTEGER,
    away_team_id  TEXT,
    away_abbr     TEXT,
    away_score    INTEGER,
    updated_at    TEXT
);
CREATE TABLE IF NOT EXISTS team_stats (
    event_id      TEXT,
    team_id       TEXT,
    team_abbr     TEXT,
    stat_name     TEXT,
    label         TEXT,
    value         REAL,
    display_value TEXT,
    PRIMARY KEY (event_id, team_id, stat_name)
);
CREATE TABLE IF NOT EXISTS player_stats (
    event_id      TEXT,
    team_id       TEXT,
    team_abbr     TEXT,
    athlete_id    TEXT,
    athlete_name  TEXT,
    category      TEXT,
    stat_key      TEXT,
    value         REAL,
    display_value TEXT,
    PRIMARY KEY (event_id, athlete_id, category, stat_key)
);
"""


def fetch_json(path, params=None, retries=3):
    url = f"{BASE}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    # Keep urllib's default User-Agent: ESPN answers 403 to browser-style and
    # made-up ones but accepts this one.
    req = urllib.request.Request(url)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def to_number(text):
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def to_int(text):
    n = to_number(text)
    return None if n is None else int(n)


def parse_weeks(spec):
    if not spec:
        return [None]
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(w) for w in spec.split(",")]


def game_row(event, now):
    comp = event["competitions"][0]
    status = event["status"]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    home, away = sides["home"], sides["away"]
    return (
        event["id"],
        event.get("season", {}).get("year"),
        event.get("season", {}).get("type"),
        event.get("week", {}).get("number"),
        event.get("date"),
        event.get("name"),
        status["type"]["name"],
        status["type"].get("detail"),
        status.get("period"),
        status.get("displayClock"),
        comp.get("venue", {}).get("fullName"),
        home["team"]["id"],
        home["team"]["abbreviation"],
        to_int(home.get("score")),
        away["team"]["id"],
        away["team"]["abbreviation"],
        to_int(away.get("score")),
        now,
    )


def team_stat_rows(event_id, boxscore):
    for team in boxscore.get("teams", []):
        t = team["team"]
        for s in team.get("statistics", []):
            yield (event_id, t["id"], t["abbreviation"], s["name"], s.get("label"),
                   to_number(s.get("value")), s.get("displayValue"))


def split_stat(key, display):
    """ESPN packs some stats into one cell ('3/5' under
    'completions/passingAttempts', '2-15' under 'sacks-sackYardsLost');
    split them into one (key, display) pair per part."""
    for sep in ("/", "-"):
        parts = key.split(sep)
        if len(parts) > 1:
            values = str(display).split(sep)
            if len(values) == len(parts):
                return list(zip(parts, values))
    return [(key, display)]


def player_stat_rows(event_id, boxscore):
    for team_block in boxscore.get("players", []):
        t = team_block["team"]
        for group in team_block.get("statistics", []):
            for athlete in group.get("athletes", []):
                a = athlete["athlete"]
                for key, display in zip(group["keys"], athlete["stats"]):
                    for k, v in split_stat(key, display):
                        yield (event_id, t["id"], t["abbreviation"], a["id"],
                               a["displayName"], group["name"], k,
                               to_number(v), v)


def ingest_week(conn, week, season, seasontype):
    params = {}
    if week is not None:
        params["week"] = week
    if season:
        params["dates"] = season
    if seasontype:
        params["seasontype"] = seasontype
    events = fetch_json("scoreboard", params)["events"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    loaded = 0
    for event in events:
        conn.execute("INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     game_row(event, now))
        if event["status"]["type"]["state"] == "pre":
            continue  # not started, no box score yet
        box = fetch_json("summary", {"event": event["id"]}).get("boxscore", {})
        conn.executemany("INSERT OR REPLACE INTO team_stats VALUES (?,?,?,?,?,?,?)",
                         team_stat_rows(event["id"], box))
        conn.executemany("INSERT OR REPLACE INTO player_stats VALUES (?,?,?,?,?,?,?,?,?)",
                         player_stat_rows(event["id"], box))
        loaded += 1
    conn.commit()
    return len(events), loaded


def export_csvs(conn):
    os.makedirs(CSV_DIR, exist_ok=True)
    for table in ("games", "team_stats", "player_stats"):
        cur = conn.execute(f"SELECT * FROM {table}")
        with open(os.path.join(CSV_DIR, f"{table}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(col[0] for col in cur.description)
            writer.writerows(cur)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weeks", help="week or range, e.g. 3 or 1-4 (default: current week)")
    ap.add_argument("--season", type=int, help="season year (default: current)")
    ap.add_argument("--seasontype", type=int, choices=(1, 2, 3),
                    help="1=preseason, 2=regular, 3=postseason (default: current)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        for week in parse_weeks(args.weeks):
            games, with_stats = ingest_week(conn, week, args.season, args.seasontype)
            label = f"week {week}" if week is not None else "current week"
            print(f"{label}: {games} games, {with_stats} with box scores")
        export_csvs(conn)
        for table in ("games", "team_stats", "player_stats"):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {n} rows")
    print(f"Database: {DB_PATH}   Power BI CSVs: {CSV_DIR}/")


if __name__ == "__main__":
    main()
