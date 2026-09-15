"""Builds the player pool the optimizer runs against: real FanDuel salaries
(from an uploaded export) merged with real performance-based projections
(from nflverse), falling back to FanDuel's own FPPG where no stats match.
"""
import os

import pandas as pd

from fanduel_import import load_fanduel_csv
from real_stats import get_offense_projections, normalize_name
from optimizer import PLAYERS_CSV

UPLOAD_PATH = "data/uploads/fanduel_latest.csv"


def save_uploaded_csv(file_storage) -> pd.DataFrame:
    """Validates and persists an uploaded FanDuel export, returning its parsed form."""
    df = load_fanduel_csv(file_storage.stream)
    os.makedirs(os.path.dirname(UPLOAD_PATH), exist_ok=True)
    df.to_csv(UPLOAD_PATH, index=False)
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

    return pool, f"uploaded FanDuel salaries + {stats_note}"
