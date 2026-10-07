"""Builds the player pool the optimizer runs against: real FanDuel salaries
(from an uploaded export) merged with real performance-based projections
(from nflverse), falling back to FanDuel's own FPPG where no stats match.
"""
import os

import pandas as pd

from fanduel_import import load_fanduel_csv
from real_stats import get_offense_projections, normalize_name
from matchup_filters import (
    build_eligibility,
    BOTTOM_N_PASS_DEFENSE,
    BOTTOM_N_RUSH_DEFENSE,
    BOTTOM_N_PASS_DEFENSE_WR_TE,
)
from optimizer import PLAYERS_CSV, ROSTER_SLOTS
from vegas_strategy import build_vegas_pool

UPLOAD_PATH = "data/uploads/fanduel_latest.csv"

STRATEGY_MATCHUP = "matchup"
STRATEGY_VEGAS = "vegas"

# Positions allowed to fall back to the full real slate (ignoring the
# matchup-eligibility rule) when too few players clear it to fill the
# roster. QB and RB are deliberately excluded -- they only need 1 and 2
# picks respectively, so the strict rule stays in force for them even when
# thin, per how this was scoped when added.
RELAXABLE_POSITIONS = ("WR", "TE", "DEF")


def _apply_matchup_eligibility(pool: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Restricts QB/RB/WR/TE to this week's matchup-eligible players (QB/RB
    are purely an opponent-defense check; WR/TE also require real personal
    production -- see matchup_filters.py for the exact rules), and DEF to
    teams facing a turnover-prone, poor-pass-protection offense. DEF is
    matched by team alone (there's one defense per team, not a roster of
    them to rank by name). Also attaches an "opponent" column (this week's
    opponent for each player's own team) so optimizer.py can keep a DEF off
    the same lineup as a QB/WR/TE it's playing against. Falls back to the
    unfiltered pool with no "opponent" column, noted as such, if the live
    schedule/stats can't be fetched.

    If the strict rule leaves fewer WR, TE, or DEF than the roster requires
    (ROSTER_SLOTS[...]), that position alone falls back to the full real
    slate instead of leaving the lineup impossible to build -- e.g. a week
    early in the season where only 2 WRs clear every filter, but the roster
    needs 3. QB and RB keep the strict requirement regardless.
    """
    try:
        result = build_eligibility()
    except Exception:
        return pool, "matchup filters unavailable this run"

    eligible = result["eligible"]
    eligible_def_teams = result["eligible_def_teams"]
    teams = pool["team"].str.upper()
    keys = list(zip(pool["name"].apply(normalize_name), teams))
    eligible_mask = pd.Series(
        [
            (team in eligible_def_teams) if pos == "DEF" else (key in eligible.get(pos, set()))
            for key, pos, team in zip(keys, pool["position"], teams)
        ],
        index=pool.index,
    )

    relaxed_positions = [
        pos for pos in RELAXABLE_POSITIONS
        if int(((pool["position"] == pos) & eligible_mask).sum()) < ROSTER_SLOTS[pos]
    ]

    keep = eligible_mask | pool["position"].isin(relaxed_positions)
    filtered = pool[keep].copy()
    filtered["opponent"] = teams[keep].map(result["matchups"])
    filtered = filtered.reset_index(drop=True)

    note = (
        f"week {result['week']} matchup filters ("
        f"bottom-{BOTTOM_N_PASS_DEFENSE} pass D for QB / bottom-{BOTTOM_N_RUSH_DEFENSE} rush D for RB / "
        f"25%+ target share + bottom-{BOTTOM_N_PASS_DEFENSE_WR_TE} pass D for WR / "
        f"scored a TD + bottom-{BOTTOM_N_PASS_DEFENSE_WR_TE} pass D for TE / "
        f"bottom-10 opponent turnovers+sacks-allowed for DEF, "
        "no injury designation"
    )
    if relaxed_positions:
        note += f"; {'/'.join(relaxed_positions)} rule relaxed this week (too few eligible)"
    note += ")"
    return filtered, note


def _refresh_sample_data(df: pd.DataFrame) -> None:
    """Keeps the bundled sample-data fallback (PLAYERS_CSV) in sync with the
    latest real FanDuel upload, so anyone who hasn't uploaded their own
    export still sees a current, real slate instead of a stale fixture.

    Uses FanDuel's own FPPG as the projection, since the real nflverse stats
    merge only happens later in build_player_pool -- good enough for a
    fallback dataset. Some rows (bench players with no games yet) have no
    FPPG at all; those become 0 rather than left blank, since a blank
    projected_points would break the optimizer's objective function.

    Some weeks, one or two teams are legitimately absent from a FanDuel
    export entirely -- e.g. a team that already played Thursday night before
    this export was pulled. That's expected, not a bug to work around here.
    """
    sample = df.rename(columns={"fppg": "projected_points"})
    sample["projected_points"] = sample["projected_points"].fillna(0).round(1)
    sample[["name", "position", "team", "salary", "projected_points"]].to_csv(
        PLAYERS_CSV, index=False
    )


def save_uploaded_csv(file_storage) -> pd.DataFrame:
    """Validates and persists an uploaded FanDuel export, returning its parsed form."""
    df = load_fanduel_csv(file_storage.stream)
    os.makedirs(os.path.dirname(UPLOAD_PATH), exist_ok=True)
    df.to_csv(UPLOAD_PATH, index=False)
    _refresh_sample_data(df)
    return df


def has_uploaded_pool() -> bool:
    return os.path.exists(UPLOAD_PATH)


def _apply_vegas_strategy(pool: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Top-3 Vegas games strategy (see vegas_strategy.py). The rule-by-rule
    picks are attached as pool.attrs["strategy_notes"] for the page to show.
    Falls back to the full-slate matchup filters if lines or stats can't be
    fetched."""
    try:
        restricted, notes = build_vegas_pool(pool)
    except Exception as exc:
        pool, note = _apply_matchup_eligibility(pool)
        return pool, f"Top-3 Vegas games strategy unavailable ({exc}); using {note}"
    restricted.attrs["strategy_notes"] = notes
    return restricted, "Top-3 Vegas games strategy, no injury designation"


def build_player_pool(
    force_refresh_stats: bool = False, strategy: str = STRATEGY_MATCHUP
) -> tuple[pd.DataFrame, str]:
    """Returns (players_df, source_description) for the optimizer.

    players_df has columns: name, position, team, salary, projected_points.
    Falls back to the bundled sample data if no FanDuel export has been uploaded.
    `strategy` picks the lineup rules for an uploaded pool: STRATEGY_MATCHUP
    (full-slate matchup filters) or STRATEGY_VEGAS (top-3 Vegas games).
    """
    if not has_uploaded_pool():
        sample = pd.read_csv(PLAYERS_CSV)
        return sample, "sample data (upload a FanDuel export to use real salaries)"

    salaries = pd.read_csv(UPLOAD_PATH)
    salaries["match_key"] = salaries["name"].apply(normalize_name)

    offense = salaries[salaries["position"] != "DEF"].copy()
    defense = salaries[salaries["position"] == "DEF"].copy()

    try:
        real_stats = get_offense_projections(force_refresh=force_refresh_stats)
        merged = offense.merge(
            real_stats[["match_key", "position", "projected_points"]],
            on=["match_key", "position"],
            how="left",
        )
        stats_note = f"real nflverse stats (season {real_stats['season'].iloc[0]})" if len(real_stats) else "FanDuel FPPG"
    except Exception:
        merged = offense.copy()
        merged["projected_points"] = pd.NA
        stats_note = "FanDuel FPPG (nflverse stats unavailable)"

    # Fall back to FanDuel's own FPPG for anyone real stats didn't match.
    merged["projected_points"] = merged["projected_points"].fillna(merged["fppg"])
    defense["projected_points"] = defense["fppg"]

    pool = pd.concat([merged, defense], ignore_index=True)
    pool = pool[["name", "position", "team", "salary", "projected_points"]]
    pool = pool.dropna(subset=["projected_points"])
    pool["projected_points"] = pool["projected_points"].round(2)

    if strategy == STRATEGY_VEGAS:
        pool, eligibility_note = _apply_vegas_strategy(pool)
    else:
        pool, eligibility_note = _apply_matchup_eligibility(pool)

    return pool, f"uploaded FanDuel salaries + {stats_note} + {eligibility_note}"
