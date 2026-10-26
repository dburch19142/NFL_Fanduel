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
from vegas_strategy import _load_week_context, build_vegas_pool
from alpha_rule import build_alpha_pool
from echo_rule import build_echo_pool
from beta_strategy import beta_notes, beta_rb_mask, load_rushing_ranks, qualifying_teams
from nfl_data import apply_injury_report

UPLOAD_PATH = "data/uploads/fanduel_latest.csv"

STRATEGY_MATCHUP = "matchup"
STRATEGY_VEGAS = "vegas"
STRATEGY_COMBINED = "combined"
STRATEGY_ALPHA = "alpha"
STRATEGY_BETA = "beta"
STRATEGY_ECHO = "echo"

# Positions allowed to fall back to the full real slate (ignoring the
# matchup-eligibility rule) when too few players clear it to fill the
# roster. QB and RB are deliberately excluded -- they only need 1 and 2
# picks respectively, so the strict rule stays in force for them even when
# thin, per how this was scoped when added.
RELAXABLE_POSITIONS = ("WR", "TE", "DEF")


def _matchup_eligible_mask(pool: pd.DataFrame, result: dict) -> pd.Series:
    """True for each row of `pool` that clears this week's matchup rule for
    its position (`result` is build_eligibility's). DEF is matched by team
    alone; everyone else by normalized name + team."""
    eligible = result["eligible"]
    eligible_def_teams = result["eligible_def_teams"]
    teams = pool["team"].str.upper()
    keys = list(zip(pool["name"].apply(normalize_name), teams))
    return pd.Series(
        [
            (team in eligible_def_teams) if pos == "DEF" else (key in eligible.get(pos, set()))
            for key, pos, team in zip(keys, pool["position"], teams)
        ],
        index=pool.index,
    )


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

    teams = pool["team"].str.upper()
    eligible_mask = _matchup_eligible_mask(pool, result)

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


def _apply_combined_strategy(pool: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Full slate with a Vegas stack: the Top-3 Vegas games rules pick the
    QB, his WR1 stack, the WR2 bring-back, RB1, RB2 and DEF (see
    vegas_strategy.py); every other slot comes from players in any game who
    clear the matchup filters (see matchup_filters.py). As with the matchup
    strategy alone, WR and TE fall back to the full slate when too few
    qualify to fill the roster. Falls back to the matchup filters by
    themselves if lines or stats can't be fetched."""
    try:
        vegas, notes = build_vegas_pool(pool, core_only=True)
        result = build_eligibility()
    except Exception as exc:
        pool, note = _apply_matchup_eligibility(pool)
        return pool, f"Combined strategy unavailable ({exc}); using {note}"

    key = ["name", "team", "position"]
    picks = vegas.loc[vegas["required_group"].notna(), key + ["required_group"]]
    combined = pool.merge(picks.drop_duplicates(key), on=key, how="left")
    is_pick = combined["required_group"].notna()
    usable = is_pick | _matchup_eligible_mask(combined, result)

    relaxed_positions = [
        pos for pos in RELAXABLE_POSITIONS
        if int(((combined["position"] == pos) & usable).sum()) < ROSTER_SLOTS[pos]
    ]
    keep = usable | combined["position"].isin(relaxed_positions)
    combined = combined[keep].copy()
    combined["opponent"] = combined["team"].str.upper().map(result["matchups"])
    combined = combined.reset_index(drop=True)

    others = combined[combined["required_group"].isna()]
    notes.append("Other slots, matchup-eligible players from any game: " + ", ".join(
        f"{int((others['position'] == pos).sum())} {pos}" for pos in ("RB", "WR", "TE")
    ) + ".")
    if relaxed_positions:
        notes.append(f"{'/'.join(relaxed_positions)} matchup rule relaxed this week (too few eligible).")
    combined.attrs["strategy_notes"] = notes
    return combined, "combined strategy (Top-3 Vegas games picks + full-slate matchup filters), no injury designation"


def _apply_alpha_strategy(pool: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Alpha Rule strategy (see alpha_rule.py). The rule-by-rule picks are
    attached as pool.attrs["strategy_notes"] for the page to show. Falls
    back to the full-slate matchup filters if lines or stats can't be
    fetched."""
    try:
        restricted, notes = build_alpha_pool(pool)
    except Exception as exc:
        pool, note = _apply_matchup_eligibility(pool)
        return pool, f"Alpha Rule unavailable ({exc}); using {note}"
    restricted.attrs["strategy_notes"] = notes
    return restricted, "Alpha Rule (top-3 Sunday 1pm and 4pm games), no injury designation"


def _apply_echo_strategy(pool: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Echo Rule strategy (see echo_rule.py): the Alpha Rule with the Beta
    strategy's RBs and seven updates for the cheap picks. Notes and fallback
    work like the Alpha Rule's."""
    try:
        restricted, notes = build_echo_pool(pool)
    except Exception as exc:
        pool, note = _apply_matchup_eligibility(pool)
        return pool, f"Echo Rule unavailable ({exc}); using {note}"
    restricted.attrs["strategy_notes"] = notes
    return restricted, "Echo Rule (Alpha Rule + Beta RBs + value, role and game-script updates), no injury designation"


def _apply_beta_strategy(pool: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Beta strategy (see beta_strategy.py): RBs only from teams with a
    top-10 rushing offense facing a bottom-10 rushing defense on Sunday at
    1pm or 4pm; every other position by the matchup filters, relaxed for
    WR/TE/DEF when too few qualify, as in the matchup strategy. Falls back to
    the matchup filters alone if lines or stats can't be fetched, and to
    matchup-eligible RBs when the rule leaves fewer than two."""
    try:
        result = build_eligibility()
        odds = _load_week_context(result["season"], result["week"])["odds"]
        qualifying = qualifying_teams(load_rushing_ranks(result["season"]), odds)
    except Exception as exc:
        pool, note = _apply_matchup_eligibility(pool)
        return pool, f"Beta strategy unavailable ({exc}); using {note}"

    notes = beta_notes(qualifying)
    beta_rb = beta_rb_mask(pool, {q["team"] for q in qualifying})
    matchup = _matchup_eligible_mask(pool, result)
    if int(beta_rb.sum()) >= ROSTER_SLOTS["RB"]:
        rb_ok = beta_rb
        notes.append("RBs on those teams: " + ", ".join(pool.loc[beta_rb, "name"]) + ".")
    else:
        rb_ok = (pool["position"] == "RB") & (beta_rb | matchup)
        notes.append(f"Only {int(beta_rb.sum())} RB(s) on qualifying teams after injuries; "
                     "RBs fall back to the matchup filters this week.")

    others = pool["position"] != "RB"
    usable = rb_ok | (others & matchup)
    relaxed_positions = [
        pos for pos in RELAXABLE_POSITIONS
        if int(((pool["position"] == pos) & usable).sum()) < ROSTER_SLOTS[pos]
    ]
    keep = usable | pool["position"].isin(relaxed_positions)
    filtered = pool[keep].copy()
    filtered["opponent"] = filtered["team"].str.upper().map(result["matchups"])
    filtered = filtered.reset_index(drop=True)
    if relaxed_positions:
        notes.append(f"{'/'.join(relaxed_positions)} matchup rule relaxed this week (too few eligible).")
    filtered.attrs["strategy_notes"] = notes
    return filtered, "Beta strategy (rushing matchup RBs + full-slate matchup filters), no injury designation"


def build_player_pool(
    force_refresh_stats: bool = False, strategy: str = STRATEGY_MATCHUP
) -> tuple[pd.DataFrame, str]:
    """Returns (players_df, source_description) for the optimizer.

    players_df has columns: name, position, team, salary, projected_points,
    and attrs["injury_exclusions"] lists who the live injury report removed
    (see nfl_data.py). Falls back to the bundled sample data if no FanDuel
    export has been uploaded.
    `strategy` picks the lineup rules for an uploaded pool: STRATEGY_MATCHUP
    (full-slate matchup filters), STRATEGY_VEGAS (top-3 Vegas games),
    STRATEGY_COMBINED (Vegas picks, matchup filters for the other slots), or
    STRATEGY_ALPHA (the Alpha Rule, top-3 Sunday 1pm and 4pm games only),
    STRATEGY_BETA (RBs from top-10 rush offenses facing bottom-10 rush
    defenses on Sunday afternoon, matchup filters for the rest), or
    STRATEGY_ECHO (the Echo Rule, the Alpha Rule with Beta RBs and seven
    updates).
    """
    if not has_uploaded_pool():
        sample, injury_note, removed = apply_injury_report(pd.read_csv(PLAYERS_CSV))
        sample.attrs["injury_exclusions"] = removed
        return sample, f"sample data (upload a FanDuel export to use real salaries) + {injury_note}"

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

    # Before the strategies, so their "injured starter" fill-in logic sees
    # this week's report rather than the FanDuel export's snapshot.
    pool, injury_note, removed = apply_injury_report(pool)

    if strategy == STRATEGY_VEGAS:
        pool, eligibility_note = _apply_vegas_strategy(pool)
    elif strategy == STRATEGY_COMBINED:
        pool, eligibility_note = _apply_combined_strategy(pool)
    elif strategy == STRATEGY_ALPHA:
        pool, eligibility_note = _apply_alpha_strategy(pool)
    elif strategy == STRATEGY_BETA:
        pool, eligibility_note = _apply_beta_strategy(pool)
    elif strategy == STRATEGY_ECHO:
        pool, eligibility_note = _apply_echo_strategy(pool)
    else:
        pool, eligibility_note = _apply_matchup_eligibility(pool)

    pool.attrs["injury_exclusions"] = removed
    return pool, f"uploaded FanDuel salaries + {stats_note} + {injury_note} + {eligibility_note}"
