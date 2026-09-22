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
    TOP_N,
    BOTTOM_N_PASS_DEFENSE,
    BOTTOM_N_RUSH_DEFENSE,
    BOTTOM_N_PASS_DEFENSE_WR_TE,
)
from optimizer import PLAYERS_CSV

UPLOAD_PATH = "data/uploads/fanduel_latest.csv"


def _apply_matchup_eligibility(pool: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Restricts QB/RB/WR/TE to this week's matchup-eligible players (see
    matchup_filters.py for the exact top-N/bottom-N rules); DEF passes
    through unfiltered, since FanDuel doesn't expose individual defensive
    players to rank the same way. Falls back to the unfiltered pool, noted
    as such, if the live schedule/stats can't be fetched.
    """
    try:
        result = build_eligibility()
    except Exception:
        return pool, "matchup filters unavailable this run"

    eligible = result["eligible"]
    keys = list(zip(pool["name"].apply(normalize_name), pool["team"].str.upper()))
    keep = [
        pos == "DEF" or key in eligible.get(pos, set())
        for key, pos in zip(keys, pool["position"])
    ]
    filtered = pool[keep].reset_index(drop=True)
    note = (
        f"week {result['week']} matchup filters (top-{TOP_N} stats, "
        f"bottom-{BOTTOM_N_PASS_DEFENSE} pass D for QB / bottom-{BOTTOM_N_RUSH_DEFENSE} rush D for RB / "
        f"bottom-{BOTTOM_N_PASS_DEFENSE_WR_TE} pass D for WR-TE, "
        "no injury designation)"
    )
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


def build_player_pool(force_refresh_stats: bool = False) -> tuple[pd.DataFrame, str]:
    """Returns (players_df, source_description) for the optimizer.

    players_df has columns: name, position, team, salary, projected_points.
    Falls back to the bundled sample data if no FanDuel export has been uploaded.
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

    pool, eligibility_note = _apply_matchup_eligibility(pool)

    return pool, f"uploaded FanDuel salaries + {stats_note} + {eligibility_note}"
