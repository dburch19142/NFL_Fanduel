"""Matchup-based lineup eligibility rules, computed from real per-game NFL
stats and the upcoming week's schedule (both pulled live from nflverse).

Rules implemented, each independently toggleable via ELIGIBILITY_RULES.
QB and RB eligibility is driven entirely by the opponent's defense this
week; WR and TE also require a real personal-production bar:
  QB:  this week's opponent must be a bottom-10 pass defense (most pass
       yards allowed/game).
  RB:  this week's opponent must be a bottom-10 rush defense in BOTH rush
       yards allowed/game AND rush TDs allowed/game.
  WR:  target share (share of the team's own targets/game) must be at least
       MIN_WR_TARGET_SHARE, AND this week's opponent must be a bottom-15
       pass defense in BOTH pass yards allowed/game and pass TDs
       allowed/game.
  TE:  must have scored at least one receiving TD this season (a proxy for
       a TD-friendly, red-zone role -- true red-zone target data isn't
       available here), AND the same bottom-15 pass-defense matchup
       requirement as WR.
  DEF: eligible only when this week's opponent offense is bottom-10 in BOTH
       interceptions thrown/game AND sacks allowed/game -- a turnover-prone,
       poor-pass-protection offense sets up easy defensive scoring. Unlike
       the other positions, this isn't about the defense's own stats (FanDuel
       doesn't expose individual defensive players to rank that way); it's
       purely about how bad the offense it's facing is.

All rankings behind these bottom-N cutoffs are per-team TOTALS across
whatever regular-season games have been played so far, divided by games
played (i.e. a true per-game average, not just a raw season total) -- so a
team that's played only one game is ranked on that one game exactly like
everyone else this early in a season. With few games played, these are
small samples: it's normal for a bottom-N cutoff to intersect with this
week's schedule to exclude most or all players at a position. That's a
real, correct outcome, not a bug to paper over.
"""
import json
import os
import time

import pandas as pd

from real_stats import normalize_name

BOTTOM_N_PASS_DEFENSE = 10
BOTTOM_N_RUSH_DEFENSE = 10
BOTTOM_N_PASS_DEFENSE_WR_TE = 15
BOTTOM_N_DEF_MATCHUP = 10
MIN_WR_TARGET_SHARE = 0.25
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
    with fair tie-handling: everyone tied for the n-th spot is included,
    rather than an arbitrary subset of that tied group being cut off to hit
    exactly `n` rows (which is what plain nsmallest/nlargest would do, and
    ties at the cutoff are common early in a season with few games played)."""
    ranks = df[stat_col].rank(method="min", ascending=False)
    return set(df.loc[ranks <= n, "team"])


def defense_allowed_per_game(season: int) -> pd.DataFrame:
    """Per-team yards/TDs allowed per game, one row per team, with columns
    team, pass_yds_allowed, pass_tds_allowed, rush_yds_allowed,
    rush_tds_allowed."""
    team_week = pd.read_parquet(STATS_TEAM_WEEK_URL.format(season))
    team_week = team_week[team_week["season_type"] == "REG"]

    per_game = _per_game(
        team_week, ["team"], ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds"]
    )
    opponent_lookup_pass_yds = per_game.set_index("team")["passing_yards"]
    opponent_lookup_pass_tds = per_game.set_index("team")["passing_tds"]
    opponent_lookup_rush_yds = per_game.set_index("team")["rushing_yards"]
    opponent_lookup_rush_tds = per_game.set_index("team")["rushing_tds"]

    # A team's defense "allows" whatever its opponents' offenses average --
    # approximated here as the average of each opponent's own per-game output,
    # by joining each week's game back to that same opponent's own row.
    allowed = team_week[["team", "opponent_team", "week"]].drop_duplicates()
    allowed["pass_yds_allowed"] = allowed["opponent_team"].map(opponent_lookup_pass_yds)
    allowed["pass_tds_allowed"] = allowed["opponent_team"].map(opponent_lookup_pass_tds)
    allowed["rush_yds_allowed"] = allowed["opponent_team"].map(opponent_lookup_rush_yds)
    allowed["rush_tds_allowed"] = allowed["opponent_team"].map(opponent_lookup_rush_tds)
    allowed = allowed.groupby("team").agg(
        pass_yds_allowed=("pass_yds_allowed", "mean"),
        pass_tds_allowed=("pass_tds_allowed", "mean"),
        rush_yds_allowed=("rush_yds_allowed", "mean"),
        rush_tds_allowed=("rush_tds_allowed", "mean"),
    ).reset_index()
    return allowed


def get_defense_eligibility(season: int) -> dict[str, set[str]]:
    """Returns team abbreviations allowing the most yards/TDs per game of
    each type, as several independently-sized bottom-N sets:
      'pass':        bottom-BOTTOM_N_PASS_DEFENSE by pass yards allowed/game (used by QB)
      'rush_yards':  bottom-BOTTOM_N_RUSH_DEFENSE by rush yards allowed/game (used by RB)
      'rush_tds':    bottom-BOTTOM_N_RUSH_DEFENSE by rush TDs allowed/game (used by RB)
      'pass_yards_wr_te': bottom-BOTTOM_N_PASS_DEFENSE_WR_TE by pass yards allowed/game
      'pass_tds_wr_te':   bottom-BOTTOM_N_PASS_DEFENSE_WR_TE by pass TDs allowed/game
    RB eligibility requires a team in *both* 'rush_yards' and 'rush_tds';
    WR/TE eligibility requires a team in *both* of the last two -- see the
    module docstring."""
    allowed = defense_allowed_per_game(season)

    worst_pass = _bottom_n_teams(allowed, "pass_yds_allowed", BOTTOM_N_PASS_DEFENSE)
    worst_rush_yards = _bottom_n_teams(allowed, "rush_yds_allowed", BOTTOM_N_RUSH_DEFENSE)
    worst_rush_tds = _bottom_n_teams(allowed, "rush_tds_allowed", BOTTOM_N_RUSH_DEFENSE)
    worst_pass_yards_wr_te = _bottom_n_teams(allowed, "pass_yds_allowed", BOTTOM_N_PASS_DEFENSE_WR_TE)
    worst_pass_tds_wr_te = _bottom_n_teams(allowed, "pass_tds_allowed", BOTTOM_N_PASS_DEFENSE_WR_TE)
    return {
        "pass": worst_pass,
        "rush_yards": worst_rush_yards,
        "rush_tds": worst_rush_tds,
        "pass_yards_wr_te": worst_pass_yards_wr_te,
        "pass_tds_wr_te": worst_pass_tds_wr_te,
    }


def get_offense_weakness(season: int) -> dict[str, set[str]]:
    """Returns team abbreviations whose own OFFENSE is worst at protecting
    the ball/QB, the inverse of get_defense_eligibility -- this is about
    who a defense is playing against, not the defense's own stats:
      'turnovers':     bottom-BOTTOM_N_DEF_MATCHUP by interceptions thrown/game
      'sacks_allowed': bottom-BOTTOM_N_DEF_MATCHUP by sacks allowed/game
    A team's DEF is matchup-eligible only when this week's opponent is in
    *both* sets -- see the module docstring."""
    team_week = pd.read_parquet(STATS_TEAM_WEEK_URL.format(season))
    team_week = team_week[team_week["season_type"] == "REG"]

    per_game = _per_game(team_week, ["team"], ["passing_interceptions", "sacks_suffered"])

    worst_turnovers = _bottom_n_teams(per_game, "passing_interceptions", BOTTOM_N_DEF_MATCHUP)
    worst_pass_pro = _bottom_n_teams(per_game, "sacks_suffered", BOTTOM_N_DEF_MATCHUP)
    return {"turnovers": worst_turnovers, "sacks_allowed": worst_pass_pro}


def _cache_path(season: int) -> str:
    return f"{CACHE_DIR}/eligibility_{season}.json"


def build_eligibility(season: int | None = None, force_refresh: bool = False) -> dict:
    """Computes this week's full eligibility picture. Returns a dict with:
      - 'season', 'week': what the eligibility was computed for
      - 'eligible': {'QB': {(name, team), ...}, 'RB': {...}, 'WR': {...}, 'TE': {...}}
        (team defenses aren't included here -- see 'eligible_def_teams' below)
      - 'eligible_def_teams': {team, ...} -- team abbreviations whose own DEF
        is matchup-eligible this week (see get_offense_weakness)
      - 'matchups': {team: opponent, ...} for this week, so callers can work
        out who a given team is playing (used for the DEF/pass-catcher
        anti-correlation rule in optimizer.py)
      - 'defense': the bottom-N defense sets described in get_defense_eligibility,
        for display/debugging

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
                "eligible_def_teams": set(raw["eligible_def_teams"]),
                "matchups": raw["matchups"],
                "defense": {k: set(v) for k, v in raw["defense"].items()},
            }

    week, matchups = get_upcoming_week_matchups(season)
    defense = get_defense_eligibility(season)
    offense_weakness = get_offense_weakness(season)

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
        ["player_display_name", "team"], ["receiving_yards", "targets", "receiving_tds", "target_share"],
    )
    te = _per_game(
        player_week[player_week["position"] == "TE"],
        ["player_display_name", "team"], ["receiving_yards", "receiving_tds"],
    )

    qb_keys = set(zip(qb["player_display_name"].apply(normalize_name), qb["team"]))
    qb_eligible = {
        (name, team) for (name, team) in qb_keys
        if matchups.get(team) in defense["pass"]
    }

    rb_defense_eligible = defense["rush_yards"] & defense["rush_tds"]
    rb_keys = set(zip(rb["player_display_name"].apply(normalize_name), rb["team"]))
    rb_eligible = {
        (name, team) for (name, team) in rb_keys
        if matchups.get(team) in rb_defense_eligible
    }

    wr_te_defense_eligible = defense["pass_yards_wr_te"] & defense["pass_tds_wr_te"]

    wr_qualified = wr[wr["target_share"] >= MIN_WR_TARGET_SHARE]
    wr_keys = set(zip(wr_qualified["player_display_name"].apply(normalize_name), wr_qualified["team"]))
    wr_eligible = {
        (name, team) for (name, team) in wr_keys
        if matchups.get(team) in wr_te_defense_eligible
    }

    # receiving_tds is a per-game average, so > 0 means "scored at least
    # once this season" -- a proxy for a TD-friendly, red-zone role.
    te_qualified = te[te["receiving_tds"] > 0]
    te_keys = set(zip(te_qualified["player_display_name"].apply(normalize_name), te_qualified["team"]))
    te_eligible = {
        (name, team) for (name, team) in te_keys
        if matchups.get(team) in wr_te_defense_eligible
    }

    eligible = {"QB": qb_eligible, "RB": rb_eligible, "WR": wr_eligible, "TE": te_eligible}

    bad_offense_teams = offense_weakness["turnovers"] & offense_weakness["sacks_allowed"]
    eligible_def_teams = {team for team, opp in matchups.items() if opp in bad_offense_teams}

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({
            "season": season,
            "week": week,
            "eligible": {pos: sorted([list(pair) for pair in s]) for pos, s in eligible.items()},
            "eligible_def_teams": sorted(eligible_def_teams),
            "matchups": matchups,
            "defense": {k: sorted(v) for k, v in defense.items()},
        }, f)

    return {
        "season": season,
        "week": week,
        "eligible": eligible,
        "eligible_def_teams": eligible_def_teams,
        "matchups": matchups,
        "defense": defense,
    }
