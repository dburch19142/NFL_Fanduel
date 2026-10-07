"""The "Top-3 Vegas games" lineup strategy: builds the whole lineup out of
the three games on the uploaded slate with the highest Vegas over/under
(games at 47+ tend to produce big QB scores), then picks or constrains each
slot by these rules:

  QB:   among each team's starting QB, those facing a bottom-10 pass defense
        (most pass yards allowed/game); of those, the best runner (rushing
        yards/game). If none face a bottom-10 pass D, the QB facing the worst
        pass D of the three games.
  WR1:  the best WR (projection) on the selected QB's team with a 25%+
        target share -- the QB stack.
  WR2:  the best WR on the QB's opponent -- the "bring-back".
  WR3:  at least one team's WR2 (second on his team in target share among
        players on the slate) from one of the two games the QB/WRs above
        didn't come from.
  RB:   at least one RB from a team starting a backup QB or facing a
        bottom-10 rush defense (rush yards allowed/game).
  TE:   only TEs targeted near the goal line (inside the 10) this season --
        TEs score mostly on touchdowns.
  FLEX: whoever the optimizer likes best.
  DEF:  the defense whose opponent is implied to score under about 20
        points, ranked next by facing a backup QB, facing a bottom-10
        sacks-allowed offense, and playing at home.

"Starting QB" is the team's season leader in pass attempts. If he's not on
the (injury-filtered) slate, the team is starting a backup: its
highest-salary QB on the slate.

Picks (QB, WR1, WR2, DEF) and at-least-one groups (WR3, RB) are handed to
the optimizer via a "required_group" column; see optimizer.py. When a rule
can't be met (e.g. no WR on the QB's team has a 25% target share), it falls
back to the closest thing and says so in the notes shown on the page.

Game lines come from ESPN's public (unofficial) scoreboard API, whose
consensus provider is DraftKings.
"""
import json
import os
import time

import pandas as pd

from espn_ingest import fetch_json
from matchup_filters import (
    BOTTOM_N_PASS_DEFENSE,
    BOTTOM_N_RUSH_DEFENSE,
    BOTTOM_N_DEF_MATCHUP,
    CACHE_DIR,
    MIN_WR_TARGET_SHARE,
    STATS_PLAYER_WEEK_URL,
    _bottom_n_teams,
    build_eligibility,
    defense_allowed_per_game,
    get_offense_weakness,
)
from real_stats import normalize_name

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{0}.parquet"

TOP_GAME_COUNT = 3
BIG_QB_TOTAL = 47.0
LOW_IMPLIED_POINTS = 20.5  # "under about 20"
GOAL_LINE_YARDS = 10
ODDS_CACHE_MAX_AGE_SECONDS = 60 * 60  # lines move; refresh hourly
STATS_CACHE_MAX_AGE_SECONDS = 6 * 60 * 60

# ESPN abbreviations that differ from nflverse's (which the pool uses).
ODDS_TEAM_ALIASES = {"WSH": "WAS", "LAR": "LA"}


# ---------------------------------------------------------------- odds

def parse_scoreboard_odds(scoreboard: dict) -> pd.DataFrame:
    """One row per game with posted odds: home, away, total, home_spread
    (negative = home favored), kickoff."""
    rows = []
    for event in scoreboard.get("events", []):
        comp = event["competitions"][0]
        odds = (comp.get("odds") or [{}])[0]
        if odds.get("overUnder") is None or odds.get("spread") is None:
            continue
        sides = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
        rows.append({
            "home": ODDS_TEAM_ALIASES.get(sides["home"], sides["home"]),
            "away": ODDS_TEAM_ALIASES.get(sides["away"], sides["away"]),
            "total": float(odds["overUnder"]),
            "home_spread": float(odds["spread"]),
            "kickoff": event.get("date"),
        })
    return pd.DataFrame(rows, columns=["home", "away", "total", "home_spread", "kickoff"])


def implied_points(game: pd.Series) -> dict[str, float]:
    """Each team's Vegas-implied score: half the total, shifted by half the spread."""
    return {
        game["home"]: (game["total"] - game["home_spread"]) / 2,
        game["away"]: (game["total"] + game["home_spread"]) / 2,
    }


def top_games(odds: pd.DataFrame, slate_teams: set[str], n: int = TOP_GAME_COUNT) -> pd.DataFrame:
    """The `n` highest-total games with both teams on the slate; ties at the
    cutoff go to the earlier kickoff."""
    on_slate = odds[odds["home"].isin(slate_teams) & odds["away"].isin(slate_teams)]
    return (
        on_slate.sort_values(["total", "kickoff"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------- rules

def _pick(pool: pd.DataFrame, mask: pd.Series, by: list[str]) -> int | None:
    candidates = pool[mask]
    if candidates.empty:
        return None
    return candidates.sort_values(by, ascending=False).index[0]


def apply_vegas_rules(
    pool: pd.DataFrame,
    odds: pd.DataFrame,
    player_stats: pd.DataFrame,
    pass_yds_allowed: dict[str, float],
    bad_pass_teams: set[str],
    bad_rush_teams: set[str],
    bad_sack_teams: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Restricts `pool` to the top-3 games and tags each rule's picks in a
    "required_group" column. Returns (pool, notes).

    player_stats columns: match_key, team, position, pass_attempts,
    rushing_yards_pg, target_share, goal_line_targets.
    """
    notes: list[str] = []
    games = top_games(odds, set(pool["team"]))
    if games.empty:
        raise ValueError("No games with Vegas lines match the uploaded slate.")

    opponent, implied, home_teams = {}, {}, set()
    for _, g in games.iterrows():
        opponent[g["home"]], opponent[g["away"]] = g["away"], g["home"]
        implied.update(implied_points(g))
        home_teams.add(g["home"])
    notes.append("Top games by over/under: " + ", ".join(
        f"{g.away} @ {g.home} ({g.total:g})" for g in games.itertuples()
    ))
    low = [f"{g.away} @ {g.home}" for g in games.itertuples() if g.total < BIG_QB_TOTAL]
    if low:
        notes.append(f"Below the {BIG_QB_TOTAL:g}+ big-QB-score total: {', '.join(low)}.")

    pool = pool[pool["team"].isin(opponent)].copy()
    pool["opponent"] = pool["team"].map(opponent)
    pool["required_group"] = None

    stats = player_stats.drop_duplicates(["match_key", "team", "position"])
    pool["match_key"] = pool["name"].apply(normalize_name)
    pool = pool.merge(stats, on=["match_key", "team", "position"], how="left")
    for col in ("pass_attempts", "rushing_yards_pg", "target_share", "goal_line_targets"):
        pool[col] = pool[col].fillna(0)

    # Starting QBs, and which teams are down to a backup.
    starters, backup_teams = {}, set()
    for team in opponent:
        leaders = stats[(stats["team"] == team) & (stats["position"] == "QB")]
        leader_key = leaders.sort_values("pass_attempts").iloc[-1]["match_key"] if len(leaders) else None
        team_qbs = pool[(pool["team"] == team) & (pool["position"] == "QB")]
        if team_qbs.empty:
            continue
        if leader_key is not None and leader_key in set(team_qbs["match_key"]):
            starters[team] = team_qbs.index[team_qbs["match_key"] == leader_key][0]
        else:
            starters[team] = team_qbs.sort_values(["salary", "projected_points"]).index[-1]
            if leader_key is not None:
                backup_teams.add(team)
    if backup_teams:
        notes.append("Starting a backup QB: " + ", ".join(
            f"{t} ({pool.loc[starters[t], 'name']})" for t in sorted(backup_teams)
        ))

    # QB
    starter_rows = pool.loc[list(starters.values())]
    vs_bad_pass = starter_rows[starter_rows["opponent"].isin(bad_pass_teams)]
    if not vs_bad_pass.empty:
        qb = vs_bad_pass.sort_values(["rushing_yards_pg", "projected_points"]).index[-1]
        why = f"best runner facing a bottom-{BOTTOM_N_PASS_DEFENSE} pass D"
    else:
        starter_rows = starter_rows.assign(
            opp_pass_allowed=starter_rows["opponent"].map(pass_yds_allowed).fillna(0)
        )
        qb = starter_rows.sort_values(["opp_pass_allowed", "rushing_yards_pg"]).index[-1]
        why = f"no starter faces a bottom-{BOTTOM_N_PASS_DEFENSE} pass D, so the one facing the worst"
    qb_team, qb_opp = pool.loc[qb, "team"], pool.loc[qb, "opponent"]
    pool.loc[qb, "required_group"] = "QB"
    notes.append(
        f"QB {pool.loc[qb, 'name']} ({qb_team} vs {qb_opp}): {why}, "
        f"{pool.loc[qb, 'rushing_yards_pg']:.0f} rush yds/game."
    )

    # WR1: stack with the QB
    wr = pool["position"] == "WR"
    wr1 = _pick(pool, wr & (pool["team"] == qb_team) & (pool["target_share"] >= MIN_WR_TARGET_SHARE),
                ["projected_points"])
    if wr1 is None:
        wr1 = _pick(pool, wr & (pool["team"] == qb_team), ["projected_points"])
        notes.append(f"No {qb_team} WR has a {MIN_WR_TARGET_SHARE:.0%}+ target share; "
                     "WR1 is simply their best WR.")
    if wr1 is not None:
        pool.loc[wr1, "required_group"] = "WR1"
        notes.append(f"WR1 {pool.loc[wr1, 'name']}: stacked with the QB "
                     f"({pool.loc[wr1, 'target_share']:.0%} target share).")

    # WR2: bring-back from the opponent
    wr2 = _pick(pool, wr & (pool["team"] == qb_opp), ["projected_points"])
    if wr2 is not None:
        pool.loc[wr2, "required_group"] = "WR2"
        notes.append(f"WR2 {pool.loc[wr2, 'name']}: best WR on the opposing team.")

    # WR3: a team's WR2 from one of the other games
    other_teams = [t for t in opponent if t not in (qb_team, qb_opp)]
    wr3_candidates = []
    for team in other_teams:
        ranked = pool[wr & (pool["team"] == team)].sort_values(
            ["target_share", "projected_points"], ascending=False
        )
        if len(ranked) >= 2:
            wr3_candidates.append(ranked.index[1])
    if wr3_candidates:
        pool.loc[wr3_candidates, "required_group"] = "WR3"
        notes.append("WR3 from the other games' WR2s: " + ", ".join(
            f"{pool.loc[i, 'name']} ({pool.loc[i, 'team']})" for i in wr3_candidates
        ))

    # RB: backup-QB team or a soft run defense
    rb_mask = (pool["position"] == "RB") & (
        pool["team"].isin(backup_teams) | pool["opponent"].isin(bad_rush_teams)
    )
    if rb_mask.any():
        pool.loc[rb_mask, "required_group"] = "RB"
        notes.append(f"At least one RB from a backup-QB team or facing a bottom-{BOTTOM_N_RUSH_DEFENSE} "
                     "rush D: " + ", ".join(sorted(set(pool.loc[rb_mask, "team"]))))
    else:
        notes.append("No RB is on a backup-QB team or facing a bottom-10 rush D; RB rule skipped.")

    # TE: goal-line targets only
    te = pool["position"] == "TE"
    goal_line_te = te & (pool["goal_line_targets"] > 0)
    if goal_line_te.any():
        pool = pool[~te | goal_line_te]
        notes.append(f"TEs limited to those targeted inside the {GOAL_LINE_YARDS}: " + ", ".join(
            pool.loc[pool["position"] == "TE", "name"]
        ))
    else:
        notes.append(f"No TE has a target inside the {GOAL_LINE_YARDS} yet; TE rule skipped.")

    # DEF: can't face the QB's team or the WR2's team (anti-correlation), so
    # it comes from the other two games.
    defs = pool[(pool["position"] == "DEF") & pool["team"].isin(other_teams)].copy()
    if not defs.empty:
        defs["opp_implied"] = defs["opponent"].map(implied)
        defs["low_implied"] = defs["opp_implied"] <= LOW_IMPLIED_POINTS
        defs["vs_backup"] = defs["opponent"].isin(backup_teams)
        defs["vs_sacks"] = defs["opponent"].isin(bad_sack_teams)
        defs["at_home"] = defs["team"].isin(home_teams)
        defs["bonus"] = defs[["vs_backup", "vs_sacks", "at_home"]].sum(axis=1)
        defs["neg_implied"] = -defs["opp_implied"]
        d = defs.sort_values(["low_implied", "bonus", "neg_implied", "projected_points"]).index[-1]
        pool.loc[d, "required_group"] = "DEF"
        met = [label for col, label in (
            ("vs_backup", "faces a backup QB"),
            ("vs_sacks", f"opponent bottom-{BOTTOM_N_DEF_MATCHUP} in sacks allowed"),
            ("at_home", "at home"),
        ) if defs.loc[d, col]]
        under = "under" if defs.loc[d, "low_implied"] else "lowest available, but not under"
        notes.append(
            f"DEF {pool.loc[d, 'name']}: opponent implied {defs.loc[d, 'opp_implied']:.1f} pts "
            f"({under} ~20)" + (f"; {', '.join(met)}" if met else "") + "."
        )

    pool = pool.drop(columns=["match_key", "pass_attempts", "rushing_yards_pg",
                              "target_share", "goal_line_targets"])
    return pool.reset_index(drop=True), notes


# ---------------------------------------------------------------- live data

def get_player_season_stats(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Per-player season numbers the rules need, from nflverse. Goal-line
    targets come from play-by-play; if that's unavailable, receiving TDs
    stand in for them."""
    cache_file = f"{CACHE_DIR}/vegas_player_stats_{season}.csv"
    if not force_refresh and os.path.exists(cache_file):
        if time.time() - os.path.getmtime(cache_file) < STATS_CACHE_MAX_AGE_SECONDS:
            return pd.read_csv(cache_file)

    weekly = pd.read_parquet(STATS_PLAYER_WEEK_URL.format(season))
    weekly = weekly[weekly["season_type"] == "REG"]
    stats = weekly.groupby(["player_id", "player_display_name", "position", "team"]).agg(
        games=("week", "nunique"),
        pass_attempts=("attempts", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        target_share=("target_share", "mean"),
        receiving_tds=("receiving_tds", "sum"),
    ).reset_index()
    stats["rushing_yards_pg"] = stats["rushing_yards"] / stats["games"]

    try:
        pbp = pd.read_parquet(PBP_URL.format(season), columns=[
            "season_type", "pass_attempt", "yardline_100", "receiver_player_id",
        ])
        near_goal = pbp[(pbp["season_type"] == "REG") & (pbp["pass_attempt"] == 1)
                        & (pbp["yardline_100"] <= GOAL_LINE_YARDS)]
        counts = near_goal.groupby("receiver_player_id").size()
        stats["goal_line_targets"] = stats["player_id"].map(counts).fillna(0)
    except Exception:
        stats["goal_line_targets"] = stats["receiving_tds"]

    stats["match_key"] = stats["player_display_name"].apply(normalize_name)
    stats = stats[["match_key", "team", "position", "pass_attempts", "rushing_yards_pg",
                   "target_share", "goal_line_targets"]]
    os.makedirs(CACHE_DIR, exist_ok=True)
    stats.to_csv(cache_file, index=False)
    return stats


def _load_week_context(season: int, week: int, force_refresh: bool = False) -> dict:
    """Odds plus team-level defense/offense rankings, cached for an hour."""
    cache_file = f"{CACHE_DIR}/vegas_context_{season}_w{week}.json"
    if not force_refresh and os.path.exists(cache_file):
        if time.time() - os.path.getmtime(cache_file) < ODDS_CACHE_MAX_AGE_SECONDS:
            with open(cache_file) as f:
                raw = json.load(f)
            raw["odds"] = pd.DataFrame(raw["odds"])
            for key in ("bad_pass_teams", "bad_rush_teams", "bad_sack_teams"):
                raw[key] = set(raw[key])
            return raw

    scoreboard = fetch_json("scoreboard", {"week": week, "dates": season, "seasontype": 2})
    odds = parse_scoreboard_odds(scoreboard)
    allowed = defense_allowed_per_game(season)
    context = {
        "odds": odds,
        "pass_yds_allowed": allowed.set_index("team")["pass_yds_allowed"].to_dict(),
        "bad_pass_teams": _bottom_n_teams(allowed, "pass_yds_allowed", BOTTOM_N_PASS_DEFENSE),
        "bad_rush_teams": _bottom_n_teams(allowed, "rush_yds_allowed", BOTTOM_N_RUSH_DEFENSE),
        "bad_sack_teams": get_offense_weakness(season)["sacks_allowed"],
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({
            **context,
            "odds": odds.to_dict(orient="records"),
            **{k: sorted(context[k]) for k in ("bad_pass_teams", "bad_rush_teams", "bad_sack_teams")},
        }, f)
    return context


def build_vegas_pool(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fetches this week's lines and stats, then applies apply_vegas_rules."""
    week_info = build_eligibility()
    season, week = week_info["season"], week_info["week"]
    context = _load_week_context(season, week)
    stats = get_player_season_stats(season)
    pool, notes = apply_vegas_rules(
        pool, context["odds"], stats, context["pass_yds_allowed"],
        context["bad_pass_teams"], context["bad_rush_teams"], context["bad_sack_teams"],
    )
    notes.insert(0, f"Week {week} lines from ESPN (DraftKings).")
    return pool, notes
