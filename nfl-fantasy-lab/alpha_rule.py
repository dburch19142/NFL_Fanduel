"""The "Alpha Rule" lineup strategy: works only from the games on the
uploaded slate kicking off in the 1pm and 4pm ET windows, and locks eight
of the nine slots by these rules:

  QB:   the starting QB of the team implied to score more in the game with
        the highest Vegas over/under.
  WR1:  that QB's WR1 -- the stack.
  WR2:  the WR1 of the team he's playing -- the "bring-back".
  WR3:  the WR1 of the team implied to score more in the game with the
        second-highest over/under.
  RB1:  the lead back facing the worst rush defense (most rush yards
        allowed/game) in these games, or one facing a team whose QB1 is out.
  RB2:  the lead back facing the second-worst rush defense, or one facing a
        team whose QB1 is out.
  DEF:  the defense facing the worst offense in these games (lowest
        Vegas-implied score); ties go to the better-projected defense.
  TE, FLEX: whoever the optimizer likes best from these games.

A team's "point total" is its Vegas-implied score (see
vegas_strategy.implied_points). A team's WR1 is its WR on the slate with the
highest target share, its lead back its best-projected RB. A team's QB1 is
out when it's starting a backup (see vegas_strategy.py). Where an RB rule
offers a choice -- the rush-defense matchup or a back facing a backup QB --
the better-projected back gets the slot.

The DEF can't share a lineup with a QB/WR/TE it's playing against (see
optimizer.py), so defenses facing the QB's, WR2's or WR3's team are passed
over.

Picks are handed to the optimizer via the "required_group" column, same as
the Top-3 Vegas games strategy. When a rule can't be met it's skipped and
the notes shown on the page say so.
"""
import pandas as pd

from matchup_filters import build_eligibility
from real_stats import normalize_name
from vegas_strategy import (
    _load_week_context,
    _pick,
    _starting_qbs,
    get_player_season_stats,
    implied_points,
    top_games,
)

KICKOFF_TIMEZONE = "America/New_York"
KICKOFF_HOURS = (13, 16)  # the 1:00 and 4:05/4:25 ET windows


def afternoon_games(odds: pd.DataFrame) -> pd.DataFrame:
    """The games kicking off in the 1pm and 4pm ET windows."""
    hours = pd.to_datetime(odds["kickoff"], utc=True).dt.tz_convert(KICKOFF_TIMEZONE).dt.hour
    return odds[hours.isin(KICKOFF_HOURS)]


def apply_alpha_rule(
    pool: pd.DataFrame,
    odds: pd.DataFrame,
    player_stats: pd.DataFrame,
    rush_yds_allowed: dict[str, float],
) -> tuple[pd.DataFrame, list[str]]:
    """Restricts `pool` to the 1pm and 4pm games and tags each rule's pick
    in a "required_group" column. Returns (pool, notes).

    player_stats columns: match_key, team, position, pass_attempts,
    target_share (others are ignored).
    """
    notes: list[str] = []
    games = top_games(afternoon_games(odds), set(pool["team"]), n=len(odds))
    if games.empty:
        raise ValueError("No 1pm or 4pm games with Vegas lines match the uploaded slate.")

    opponent, implied = {}, {}
    for _, g in games.iterrows():
        opponent[g["home"]], opponent[g["away"]] = g["away"], g["home"]
        implied.update(implied_points(g))
    notes.append("1pm and 4pm games by over/under: " + ", ".join(
        f"{g.away} @ {g.home} ({g.total:g})" for g in games.itertuples()
    ))

    def favorite(game: pd.Series) -> str:
        # The team implied to score more; a pick'em goes to the home team.
        return max((game["home"], game["away"]), key=implied.get)

    pool = pool[pool["team"].isin(opponent)].copy()
    pool["opponent"] = pool["team"].map(opponent)
    pool["required_group"] = None

    stats = player_stats.drop_duplicates(["match_key", "team", "position"])
    pool["match_key"] = pool["name"].apply(normalize_name)
    pool = pool.merge(
        stats[["match_key", "team", "position", "pass_attempts", "target_share"]],
        on=["match_key", "team", "position"], how="left",
    )
    for col in ("pass_attempts", "target_share"):
        pool[col] = pool[col].fillna(0)

    starters, backup_teams = _starting_qbs(pool, stats, opponent)
    if backup_teams:
        notes.append("QB1 out, starting a backup: " + ", ".join(
            f"{t} ({pool.loc[starters[t], 'name']})" for t in sorted(backup_teams)
        ))

    # QB
    qb_team = favorite(games.iloc[0])
    if qb_team not in starters:
        notes.append(f"No {qb_team} QB on the slate; the QB comes from {opponent[qb_team]} instead.")
        qb_team = opponent[qb_team]
    if qb_team not in starters:
        raise ValueError("no QB on the slate in the highest over/under game")
    qb, qb_opp = starters[qb_team], opponent[qb_team]
    pool.loc[qb, "required_group"] = "QB"
    notes.append(
        f"QB {pool.loc[qb, 'name']} ({qb_team} vs {qb_opp}): implied {implied[qb_team]:.1f} pts "
        "in the highest over/under game."
    )

    # WR1 stack, WR2 bring-back, WR3 from the second-highest over/under game
    stacked_offenses = {qb_team}

    def tag_wr1(team: str, label: str, why: str) -> None:
        i = _pick(pool, (pool["position"] == "WR") & (pool["team"] == team),
                  ["target_share", "projected_points"])
        if i is None:
            notes.append(f"No {team} WR on the slate; {label} rule skipped.")
            return
        pool.loc[i, "required_group"] = label
        stacked_offenses.add(team)
        notes.append(f"{label} {pool.loc[i, 'name']}: {why} "
                     f"({pool.loc[i, 'target_share']:.0%} target share).")

    tag_wr1(qb_team, "WR1", "the QB's WR1")
    tag_wr1(qb_opp, "WR2", f"WR1 of the opposing team ({qb_opp})")
    if len(games) > 1:
        wr3_team = favorite(games.iloc[1])
        tag_wr1(wr3_team, "WR3", f"WR1 of {wr3_team}, implied {implied[wr3_team]:.1f} pts "
                                 "in the second-highest over/under game")
    else:
        notes.append("Only one 1pm or 4pm game on the slate; WR3 rule skipped.")

    # RB1/RB2: lead backs facing the worst and second-worst rush defenses,
    # or one facing a team whose QB1 is out.
    rb = pool["position"] == "RB"
    lead_backs = {
        team: backs.sort_values(["projected_points", "salary"]).index[-1]
        for team, backs in pool[rb].groupby("team")
    }
    worst_rush = sorted((t for t in opponent if t in rush_yds_allowed),
                        key=rush_yds_allowed.get, reverse=True)
    for rank, label in enumerate(("RB1", "RB2")):
        targets = worst_rush[rank:rank + 1] + sorted(backup_teams)
        candidates = [
            lead_backs[opponent[t]] for t in targets
            if opponent[t] in lead_backs and pd.isna(pool.loc[lead_backs[opponent[t]], "required_group"])
        ]
        if not candidates:
            notes.append(f"No lead back fits the {label} matchup; {label} rule skipped.")
            continue
        i = pool.loc[candidates].sort_values(["projected_points", "salary"]).index[-1]
        pool.loc[i, "required_group"] = label
        faced = pool.loc[i, "opponent"]
        if faced in worst_rush[rank:rank + 1]:
            why = (f"faces the {'second-' if rank else ''}worst rush D "
                   f"({rush_yds_allowed[faced]:.0f} rush yds allowed/game)")
        else:
            why = "faces a team whose QB1 is out"
        notes.append(f"{label} {pool.loc[i, 'name']} ({pool.loc[i, 'team']} vs {faced}): {why}.")

    # DEF: facing the worst offense, outside the lineup's own stacks
    defs = pool[(pool["position"] == "DEF") & ~pool["opponent"].isin(stacked_offenses)].copy()
    if defs.empty:
        notes.append("No defense is free of the lineup's own QB/WR stacks; DEF rule skipped.")
    else:
        defs["neg_implied"] = -defs["opponent"].map(implied)
        d = defs.sort_values(["neg_implied", "projected_points"]).index[-1]
        pool.loc[d, "required_group"] = "DEF"
        faced = pool.loc[d, "opponent"]
        worst = min(implied, key=implied.get)
        scope = "worst offense" if faced == worst else (
            f"worst offense not already in the lineup ({worst} is)"
        )
        notes.append(f"DEF {pool.loc[d, 'name']}: faces {faced}, the {scope}, "
                     f"implied {implied[faced]:.1f} pts.")

    pool = pool.drop(columns=["match_key", "pass_attempts", "target_share"])
    return pool.reset_index(drop=True), notes


def build_alpha_pool(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fetches this week's lines and stats, then applies apply_alpha_rule."""
    week_info = build_eligibility()
    season, week = week_info["season"], week_info["week"]
    context = _load_week_context(season, week)
    stats = get_player_season_stats(season)
    pool, notes = apply_alpha_rule(pool, context["odds"], stats, context["rush_yds_allowed"])
    notes.insert(0, f"Week {week} lines from ESPN (DraftKings).")
    return pool, notes
