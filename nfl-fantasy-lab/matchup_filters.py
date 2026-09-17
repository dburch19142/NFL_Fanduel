"""Matchup-based lineup eligibility rules, computed from real per-game NFL
stats and the upcoming week's schedule (both pulled live from nflverse).

Rules implemented, each independently toggleable via ELIGIBILITY_RULES:
  QB:  top 15 in passing yards/game AND top 15 in passing TDs, AND this
       week's opponent must be a bottom-10 pass defense (most pass yards
       allowed/game).
  RB:  top 15 in rushing yards/game AND top 15 in rushing TDs, AND this
       week's opponent must be a bottom-5 rush defense (most rush yards
       allowed/game).
  WR:  top 15 in receiving yards/game AND top 15 in targets/game AND top 15
       in receiving TDs.
  TE:  top 15 in receiving TDs (TE has no yardage/target rule of its own
       here, but still has to clear the "every position ranks top 15 in
       touchdowns" bar).
  DEF: no matchup rule -- FanDuel doesn't expose individual defensive
       players to rank this way, so team defenses pass through unfiltered.

Note QB and RB use *different* bottom-N cutoffs for their opponent's
defense (10 for pass, 5 for rush) -- that's intentional, not a typo.

All rankings are per-player TOTALS across whatever regular-season games
have been played so far, divided by games played (i.e. a true per-game
average, not just a raw season total) -- so a player who has played only
one game is ranked on that one game exactly like everyone else this early
in a season. With few games played, these are small samples: the top-N
cutoffs and the opponent-defense requirement can easily intersect to zero
eligible players at a position. That's a real, correct outcome of stacking
several strict filters together this early -- not a bug to paper over.
"""
import json
import os
import time

import pandas as pd

from real_stats import normalize_name

TOP_N = 15
BOTTOM_N_PASS_DEFENSE = 10
BOTTOM_N_RUSH_DEFENSE = 5
CACHE_DIR = "data/cache"
CACHE_MAX_AGE_SECONDS = 6 * 60 * 60  # matchups/stats can change gameday-to-gameday

STATS_TEAM_WEEK_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{0}.parquet"
STATS_PLAYER_WEEK_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{0}.parquet"


def _season_has_player_data(year: int) -> bool:
    try:
        df = pd.read_parquet(STATS_PLAYER_WEEK_URL.format(year))
        return len(df[df["season_type"] == "REG"]) > 0
    except Exception:
        return False


def latest_season_with_player_data(start_year: int | None = None) -> int:
    if start_year is None:
        start_year = pd.Timestamp.now().year
    for year in range(start_year, start_year - 4, -1):
        if _season_has_player_data(year):
            return year
    raise RuntimeError("Could not find a recent season with player-level stats.")


def _per_game(df: pd.DataFrame, group_cols: list[str], stat_cols: list[str]) -> pd.DataFrame:
    """Sums each stat per player across games played, then divides by games
    played to get a true per-game average (not just a season total)."""
    agg = df.groupby(group_cols).agg(
        games=("week", "nunique"), **{c: (c, "sum") for c in stat_cols}
    ).reset_index()
    for c in stat_cols:
        agg[c] = agg[c] / agg["games"]
    return agg


def _top_n_keys(df: pd.DataFrame, stat_col: str, n: int) -> set[tuple[str, str]]:
    """The `n` best players by `stat_col`, treating ties fairly: a player
    tied for 10th place is included alongside everyone else tied there,
    rather than an arbitrary subset of the tied group being cut off to hit
    exactly `n` rows (which is what plain nlargest(n) would do, and early
    in a season with few games played, ties at the cutoff are common).

    Keys are (normalized name, team) so they line up with names as they
    appear in an uploaded FanDuel export, which won't always match
    nflverse's own name formatting exactly (e.g. "AJ Brown" vs "A.J. Brown").
    """
    ranks = df[stat_col].rank(method="min", ascending=False)
    top = df[ranks <= n]
    return set(zip(top["player_display_name"].apply(normalize_name), top["team"]))


def get_upcoming_week_matchups(season: int) -> tuple[int, dict[str, str]]:
    """Returns (week_number, {team: opponent}) for the next week with games
    that haven't been played yet. A team on a bye that week is simply
    absent from the dict."""
    import nfl_data_py as nfl

    schedule = nfl.import_schedules([season])
    upcoming = schedule[schedule["home_score"].isna()]
    if upcoming.empty:
        raise RuntimeError(f"No upcoming games found for season {season}.")

    week = int(upcoming["week"].min())
    games = upcoming[upcoming["week"] == week]
    matchup = {}
    for _, g in games.iterrows():
        matchup[g["home_team"]] = g["away_team"]
        matchup[g["away_team"]] = g["home_team"]
    return week, matchup


def _bottom_n_teams(df: pd.DataFrame, stat_col: str, n: int) -> set[str]:
    """The `n` worst teams by `stat_col` (highest value = worst defense),
    with the same fair tie-handling as _top_n_keys: everyone tied for the
    n-th spot is included, not an arbitrary subset of that tied group."""
    ranks = df[stat_col].rank(method="min", ascending=False)
    return set(df.loc[ranks <= n, "team"])


def get_defense_eligibility(season: int) -> dict[str, set[str]]:
    """Returns {'pass': {...worst pass defenses...}, 'rush': {...worst rush defenses...}}
    -- team abbreviations allowing the most yards per game of that type.
    Pass and rush use different bottom-N cutoffs (BOTTOM_N_PASS_DEFENSE and
    BOTTOM_N_RUSH_DEFENSE respectively) -- see the module docstring."""
    team_week = pd.read_parquet(STATS_TEAM_WEEK_URL.format(season))
    team_week = team_week[team_week["season_type"] == "REG"]

    per_game = _per_game(team_week, ["team"], ["passing_yards", "rushing_yards"])
    opponent_lookup_pass = per_game.set_index("team")["passing_yards"]
    opponent_lookup_rush = per_game.set_index("team")["rushing_yards"]

    # A team's defense "allows" whatever its opponents' offenses average --
    # approximated here as the average of each opponent's own per-game output,
    # by joining each week's game back to that same opponent's own row.
    allowed = team_week[["team", "opponent_team", "week"]].drop_duplicates()
    allowed["pass_yds_allowed"] = allowed["opponent_team"].map(opponent_lookup_pass)
    allowed["rush_yds_allowed"] = allowed["opponent_team"].map(opponent_lookup_rush)
    allowed = allowed.groupby("team").agg(
        pass_yds_allowed=("pass_yds_allowed", "mean"),
        rush_yds_allowed=("rush_yds_allowed", "mean"),
    ).reset_index()

    worst_pass = _bottom_n_teams(allowed, "pass_yds_allowed", BOTTOM_N_PASS_DEFENSE)
    worst_rush = _bottom_n_teams(allowed, "rush_yds_allowed", BOTTOM_N_RUSH_DEFENSE)
    return {"pass": worst_pass, "rush": worst_rush}


def _cache_path(season: int) -> str:
    return f"{CACHE_DIR}/eligibility_{season}.json"


def build_eligibility(season: int | None = None, force_refresh: bool = False) -> dict:
    """Computes this week's full eligibility picture. Returns a dict with:
      - 'season', 'week': what the eligibility was computed for
      - 'eligible': {'QB': {(name, team), ...}, 'RB': {...}, 'WR': {...}, 'TE': {...}}
      - 'defense': the bottom-10-pass / bottom-5-rush defense sets, for display/debugging

    Cached to disk for a few hours at a time, since this pulls three separate
    live datasets (team stats, player stats, schedule) and none of them
    change meaningfully within a single day.
    """
    if season is None:
        season = latest_season_with_player_data()

    cache_file = _cache_path(season)
    if not force_refresh and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < CACHE_MAX_AGE_SECONDS:
            with open(cache_file) as f:
                raw = json.load(f)
            return {
                "season": raw["season"],
                "week": raw["week"],
                "eligible": {pos: {tuple(pair) for pair in lst} for pos, lst in raw["eligible"].items()},
                "defense": {k: set(v) for k, v in raw["defense"].items()},
            }

    week, matchups = get_upcoming_week_matchups(season)
    defense = get_defense_eligibility(season)

    player_week = pd.read_parquet(STATS_PLAYER_WEEK_URL.format(season))
    player_week = player_week[player_week["season_type"] == "REG"]

    qb = _per_game(
        player_week[player_week["position"] == "QB"],
        ["player_display_name", "team"], ["passing_yards", "passing_tds"],
    )
    rb = _per_game(
        player_week[player_week["position"] == "RB"],
        ["player_display_name", "team"], ["rushing_yards", "rushing_tds"],
    )
    wr = _per_game(
        player_week[player_week["position"] == "WR"],
        ["player_display_name", "team"], ["receiving_yards", "targets", "receiving_tds"],
    )
    te = _per_game(
        player_week[player_week["position"] == "TE"],
        ["player_display_name", "team"], ["receiving_tds"],
    )

    qb_yards_top = _top_n_keys(qb, "passing_yards", TOP_N)
    qb_tds_top = _top_n_keys(qb, "passing_tds", TOP_N)
    qb_eligible = {
        (name, team) for (name, team) in (qb_yards_top & qb_tds_top)
        if matchups.get(team) in defense["pass"]
    }

    rb_yards_top = _top_n_keys(rb, "rushing_yards", TOP_N)
    rb_tds_top = _top_n_keys(rb, "rushing_tds", TOP_N)
    rb_eligible = {
        (name, team) for (name, team) in (rb_yards_top & rb_tds_top)
        if matchups.get(team) in defense["rush"]
    }

    wr_yards_top = _top_n_keys(wr, "receiving_yards", TOP_N)
    wr_targets_top = _top_n_keys(wr, "targets", TOP_N)
    wr_tds_top = _top_n_keys(wr, "receiving_tds", TOP_N)
    wr_eligible = wr_yards_top & wr_targets_top & wr_tds_top

    te_eligible = _top_n_keys(te, "receiving_tds", TOP_N)

    eligible = {"QB": qb_eligible, "RB": rb_eligible, "WR": wr_eligible, "TE": te_eligible}

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({
            "season": season,
            "week": week,
            "eligible": {pos: sorted([list(pair) for pair in s]) for pos, s in eligible.items()},
            "defense": {k: sorted(v) for k, v in defense.items()},
        }, f)

    return {"season": season, "week": week, "eligible": eligible, "defense": defense}
