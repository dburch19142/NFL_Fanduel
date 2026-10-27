"""The "Delta Rule" lineup strategy, built from the traits the Week 5, 6 and
7, 2026 winning lineups all shared. Instead of picking slot by slot, it
narrows the pool and sets the requirements, and the optimizer takes the
best-projected lineup that meets all of them, from the games kicking off
Sunday at 1pm or 4pm ET:

  QB:         a team's starter (its highest-priced QB on the slate) in a
              game with an over/under of 46+. No other QB stays in the pool.
  Bring-back: at least one RB, WR or TE from the team the QB is playing.
  RBs:        workhorses only -- 70%+ of their team's RB carries this
              season -- so both RB slots (and an RB at FLEX) are workhorses.
  WR1:        at least one WR who is his team's WR1 with a 25%+ target
              share.
  Low total:  at least one RB, WR or TE from a game with an over/under
              under 40.
  DEF:        underdogs only (the team implied to score less).

The optimizer's base rules still apply: the QB with a WR or TE teammate,
and no DEF facing the lineup's QB, WR or TE. No more than four players come
from one game. A requirement nobody on the slate meets is dropped, and the
notes say so; with fewer than two workhorse backs, every RB stays.

Shares and carries are this season's (see vegas_strategy.get_player_season_stats);
a team's WR1 is its healthy WR with the highest target share.
"""
import pandas as pd

from alpha_rule import afternoon_games
from echo_rule import carry_shares
from matchup_filters import build_eligibility
from optimizer import FLEX_ELIGIBLE, ROSTER_SLOTS
from real_stats import normalize_name
from vegas_strategy import _load_week_context, get_player_season_stats, implied_points

QB_MIN_TOTAL = 46.0
WORKHORSE_CARRY_SHARE = 0.70
WR1_MIN_SHARE = 0.25
LOW_TOTAL = 40.0
MAX_FROM_GAME = 4


def apply_delta_rule(
    pool: pd.DataFrame,
    odds: pd.DataFrame,
    player_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Restricts `pool` to the Sunday 1pm and 4pm games and the Delta Rule's
    players, and adds the "required_groups" and "bring_back" columns the
    optimizer enforces. Returns (pool, notes).

    player_stats columns: match_key, team, position, carries, target_share.
    """
    games = afternoon_games(odds)
    games = games[games["home"].isin(pool["team"]) & games["away"].isin(pool["team"])]
    if games.empty:
        raise ValueError("No Sunday 1pm or 4pm games with Vegas lines match the uploaded slate.")
    implied, opponent, total = {}, {}, {}
    for _, g in games.iterrows():
        implied.update(implied_points(g))
        opponent[g["home"]], opponent[g["away"]] = g["away"], g["home"]
        total[g["home"]] = total[g["away"]] = g["total"]

    def games_where(test) -> str:
        return ", ".join(f"{g.away} @ {g.home} ({g.total:g})" for g in games.itertuples() if test(g.total)) or "none"

    pool = pool[pool["team"].isin(opponent)].copy()
    pool["opponent"] = pool["team"].map(opponent)
    stats = player_stats.drop_duplicates(["match_key", "team", "position"])
    pool["match_key"] = pool["name"].apply(normalize_name)
    pool = pool.merge(stats[["match_key", "team", "position", "target_share"]],
                      on=["match_key", "team", "position"], how="left")
    pool = pool.merge(carry_shares(stats), on=["match_key", "team"], how="left")
    pool[["target_share", "carry_share"]] = pool[["target_share", "carry_share"]].fillna(0)
    notes: list[str] = []

    # QB: each team's starter, in a 46+ game.
    qbs = pool[pool["position"] == "QB"]
    starters = qbs.sort_values(["salary", "projected_points"]).groupby("team").tail(1).index
    qb_ok = pool.index.isin(starters) & (pool["team"].map(total) >= QB_MIN_TOTAL)
    if not qb_ok.any():
        notes.append(f"No QB in a {QB_MIN_TOTAL:g}+ game; any starter can be the QB.")
        qb_ok = pool.index.isin(starters)
    else:
        notes.append(f"QB: a starter from a {QB_MIN_TOTAL:g}+ game -- "
                     + games_where(lambda t: t >= QB_MIN_TOTAL) + ".")
    pool = pool[(pool["position"] != "QB") | qb_ok]

    # RBs: workhorses only.
    rb = pool["position"] == "RB"
    workhorse = rb & (pool["carry_share"] >= WORKHORSE_CARRY_SHARE)
    if int(workhorse.sum()) >= ROSTER_SLOTS["RB"]:
        notes.append(f"RBs: workhorses with {WORKHORSE_CARRY_SHARE:.0%}+ of their team's carries -- " + ", ".join(
            f"{r.name} ({r.team}, {r.carry_share:.0%})"
            for r in pool[workhorse].sort_values("carry_share", ascending=False).itertuples()) + ".")
        pool = pool[~rb | workhorse]
    else:
        notes.append(f"Fewer than two RBs have {WORKHORSE_CARRY_SHARE:.0%}+ of their team's carries; every RB stays.")

    # DEF: underdogs only.
    dog = (pool["position"] == "DEF") & (pool["team"].map(implied) < pool["opponent"].map(implied))
    if dog.any():
        notes.append("DEF: underdogs -- " + ", ".join(
            f"{r.name} (+{implied[r.opponent] - implied[r.team]:g} vs {r.opponent})"
            for r in pool[dog].itertuples()) + ".")
        pool = pool[(pool["position"] != "DEF") | dog]
    else:
        notes.append("No underdog defense on the slate; any DEF can be picked.")

    # Requirement groups: a 25%+ team WR1, and a player from a sub-40 game.
    wrs = pool[pool["position"] == "WR"].sort_values(["target_share", "projected_points"], ascending=False)
    wr1 = wrs.groupby("team").head(1)
    wr1 = wr1[wr1["target_share"].round(2) >= WR1_MIN_SHARE].index
    low = pool.index[pool["position"].isin(FLEX_ELIGIBLE) & (pool["team"].map(total) < LOW_TOTAL)]
    groups = {i: [] for i in pool.index}
    for label, idx, what in (
        ("WR1_25", wr1, f"a team WR1 with a {WR1_MIN_SHARE:.0%}+ target share"),
        ("LOW_TOTAL", low, f"an RB, WR or TE from a game under {LOW_TOTAL:g} -- "
         + games_where(lambda t: t < LOW_TOTAL)),
    ):
        if len(idx) == 0:
            notes.append(f"Nobody qualifies as {what.split(' -- ')[0]}; that requirement is dropped.")
            continue
        for i in idx:
            groups[i].append(label)
        notes.append(f"Requires {what}" + (
            ": " + ", ".join(f"{pool.loc[i, 'name']} ({pool.loc[i, 'team']}, {pool.loc[i, 'target_share']:.0%})"
                             for i in idx) if label == "WR1_25" else "") + ".")
    pool["required_groups"] = pd.Series({i: "|".join(v) for i, v in groups.items()})
    pool["bring_back"] = True
    notes.append("Requires a bring-back: an RB, WR or TE from the QB's opponent.")

    pool["max_from_game"] = MAX_FROM_GAME
    pool = pool.drop(columns=["match_key", "target_share", "carry_share"])
    return pool.reset_index(drop=True), notes


def build_delta_pool(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fetches this week's lines and stats, then applies apply_delta_rule."""
    week_info = build_eligibility()
    season, week = week_info["season"], week_info["week"]
    context = _load_week_context(season, week)
    pool, notes = apply_delta_rule(pool, context["odds"], get_player_season_stats(season))
    notes.insert(0, f"Week {week} lines from ESPN (DraftKings).")
    return pool, notes
