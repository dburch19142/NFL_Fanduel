"""The "Echo Rule" lineup strategy: the Alpha Rule (see alpha_rule.py) with
seven updates aimed at the cheap picks -- the players who have to outscore
their salary for a lineup to win. The Alpha Rule still picks QB, WR1, WR2,
RB1, RB2, WR3, TE and DEF; the updates change the games it works from, the
backs it can take, and how FLEX is filled:

  1. Injury opportunity: when a team's leader at a position is out (most
        carries for RB, highest target share for WR and TE), the player
        inheriting the role is a FLEX candidate even without a season-long
        role of his own, and an RB fill-in stays in the pool past rule 7.
  2. Role over last week: other FLEX candidates need a real role -- WRs
        and TEs a 15%+ target share, RBs 40%+ of their team's carries.
  3. Game environment: only games with an over/under of 46+ count. If
        fewer than two of the Sunday 1pm and 4pm games clear it, the rule
        relaxes to the Alpha Rule's top three.
  4. Game script: a FLEX RB must be on a team that's favored or implied
        for 24+ (teams that lead run the clock out); a FLEX WR or TE can't
        be on a team favored by 7+ (it won't need to throw).
  5. Stack correlation: besides the Alpha Rule's FLEX pool (the second and
        third games' WR2s, top two RBs and top TEs), FLEX can take the QB's
        own WR2 and TE (a second stack piece) or the opposing WR2 (a second
        bring-back), while the game has room under the four-player cap.
  6. Value: a FLEX candidate must project 2.5+ FanDuel points per $1,000
        of salary. If fewer than two clear it, the four best values stay.
        The same goes for the RB, WR and TE rule picks under $6,000: one
        below the line gives way to a better value at its position with a
        real role (rule 2's bar) -- from its own game first, then any of
        these games; one clearing 2.5 if there is one, else the best value
        -- that fits the cap and the per-game limit. RB fill-ins for an
        injured lead back (rule 1), DEF and QB are left as picked.
  7. Avoid traps: RBs with under 35% of their team's carries (committee
        backs) are dropped before the Alpha Rule runs, unless they're
        filling in for an injured lead back. Projections are season
        averages, so one big week doesn't carry a player; the injury report
        has already removed anyone with a designation, questionable
        included; and rule 3 keeps out the lowest-total games.

Notes from the Alpha Rule are kept, except its FLEX note, which the Echo
Rule's replaces.
"""
import pandas as pd

from alpha_rule import (
    ALPHA_GAME_COUNT,
    MAX_FROM_GAME,
    _leader_out_teams,
    afternoon_games,
    apply_alpha_rule,
    get_defense_vs_receivers,
)
from matchup_filters import build_eligibility
from optimizer import DEFAULT_SALARY_CAP, FLEX_COUNT, FLEX_ELIGIBLE, ROSTER_SLOTS
from real_stats import normalize_name
from vegas_strategy import _load_week_context, get_player_season_stats, implied_points, top_games

ECHO_MIN_TOTAL = 46.0
ECHO_MIN_GAMES = 2
COMMITTEE_CARRY_SHARE = 0.35
FLEX_MIN_TARGET_SHARE = 0.15
FLEX_MIN_CARRY_SHARE = 0.40
FLEX_RB_MIN_IMPLIED = 24.0
BIG_FAVORITE = 7.0  # implied-points edge where a team stops needing to throw
VALUE_FLOOR = 2.5  # FanDuel points per $1,000 of salary
VALUE_FALLBACK_COUNT = 4
CHEAP_SALARY = 6000


def value(row: pd.Series) -> float:
    """Projected FanDuel points per $1,000 of salary."""
    return row["projected_points"] / (row["salary"] / 1000) if row["salary"] else 0.0


def carry_shares(stats: pd.DataFrame) -> pd.DataFrame:
    """Each RB's share of his team's RB carries this season, keyed by
    match_key and team."""
    rbs = stats[stats["position"] == "RB"].drop_duplicates(["match_key", "team"])
    team_carries = rbs.groupby("team")["carries"].transform("sum")
    share = (rbs["carries"] / team_carries).where(team_carries > 0)
    return rbs.assign(carry_share=share)[["match_key", "team", "carry_share"]]


def echo_games(odds: pd.DataFrame, slate_teams: set[str]) -> tuple[pd.DataFrame, str]:
    """Update 3: the odds the Alpha Rule should work from, and a note."""
    rich = odds[odds["total"] >= ECHO_MIN_TOTAL]
    if len(top_games(afternoon_games(rich), slate_teams, n=ALPHA_GAME_COUNT)) >= ECHO_MIN_GAMES:
        return rich, f"Game environment: only Sunday 1pm and 4pm games with an over/under of {ECHO_MIN_TOTAL:g}+."
    return odds, (f"Game environment: fewer than {ECHO_MIN_GAMES} Sunday 1pm and 4pm games have an "
                  f"over/under of {ECHO_MIN_TOTAL:g}+; using the top {ALPHA_GAME_COUNT} instead.")


def apply_echo_rule(
    pool: pd.DataFrame,
    odds: pd.DataFrame,
    player_stats: pd.DataFrame,
    rush_yds_allowed: dict[str, float],
    pass_yds_allowed: dict[str, float],
    defense_vs: dict | None = None,
    salary_cap: float = DEFAULT_SALARY_CAP,
) -> tuple[pd.DataFrame, list[str]]:
    """The Alpha Rule plus the seven updates in the module docstring.
    Arguments and return value match apply_alpha_rule's."""
    stats = player_stats.drop_duplicates(["match_key", "team", "position"])
    slate_teams = set(pool["team"])
    odds, games_note = echo_games(odds, slate_teams)
    notes = [games_note]

    # Update 7: drop committee backs, keeping fill-ins for injured lead backs.
    keyed = pool.assign(match_key=pool["name"].apply(normalize_name))
    rb_out_teams = _leader_out_teams(keyed, stats, slate_teams, "RB", "carries")
    keyed = keyed.merge(carry_shares(stats), on=["match_key", "team"], how="left")
    committee = ((keyed["position"] == "RB") & ~keyed["team"].isin(rb_out_teams)
                 & keyed["team"].isin(set(stats.loc[(stats["position"] == "RB") & (stats["carries"] > 0), "team"]))
                 & (keyed["carry_share"].fillna(0) < COMMITTEE_CARRY_SHARE))
    if committee.any():
        notes.append(f"Committee backs dropped (under {COMMITTEE_CARRY_SHARE:.0%} of team carries): "
                     + ", ".join(keyed.loc[committee, "name"]) + ".")
    pool = pool[~committee.to_numpy()].reset_index(drop=True)

    pool, alpha_notes = apply_alpha_rule(
        pool, odds, player_stats, rush_yds_allowed, pass_yds_allowed, defense_vs, salary_cap,
    )
    notes += [n for n in alpha_notes
              if not n.startswith("FLEX") and "FLEX is the optimizer's choice" not in n]

    games = top_games(afternoon_games(odds), slate_teams, n=ALPHA_GAME_COUNT)
    implied, game_of = {}, {}
    for _, g in games.iterrows():
        implied.update(implied_points(g))
        game_of[g["home"]] = game_of[g["away"]] = f"{g['away']} @ {g['home']}"

    pool["required_group"] = pool["required_group"].where(pool["required_group"] != "FLEX")
    pool["match_key"] = pool["name"].apply(normalize_name)
    pool = pool.merge(stats[["match_key", "team", "position", "target_share", "carries"]],
                      on=["match_key", "team", "position"], how="left")
    pool = pool.merge(carry_shares(stats), on=["match_key", "team"], how="left")
    pool["target_share"] = pool["target_share"].fillna(0)
    pool["carry_share"] = pool["carry_share"].fillna(0)
    pool["value"] = pool.apply(value, axis=1)

    def has_role(rows: pd.DataFrame) -> pd.Series:
        # Update 2's bar: a 15%+ target share, or 40%+ of the carries for an RB.
        return rows["carry_share"].where(rows["position"] == "RB", rows["target_share"]) >= rows[
            "position"].map(lambda p: FLEX_MIN_CARRY_SHARE if p == "RB" else FLEX_MIN_TARGET_SHARE)

    # Update 6, rule picks: a RB, WR or TE pick under $6,000 that projects
    # below the value floor gives way to a better value at its position with
    # a real role -- from its own game first, then any of these games; one
    # clearing the floor if there is one, else the best value -- as long as
    # it fits the cap and the per-game limit.
    roster_size = sum(ROSTER_SLOTS.values()) + FLEX_COUNT
    cheapest_flex = pool.loc[pool["position"].isin(FLEX_ELIGIBLE), "salary"].min()
    # A back filling in for an injured lead back (update 1) is exempt: his
    # projection reflects his old role, not the carries he's inheriting.
    # Receivers aren't: an injured WR1's targets spread across the roster.
    rb_out = _leader_out_teams(pool, stats, set(implied), "RB", "carries")
    filling_in = (pool["position"] == "RB") & pool["team"].isin(rb_out)
    low = pool.index[pool["required_group"].notna() & pool["position"].isin(FLEX_ELIGIBLE) & ~filling_in
                     & (pool["salary"] < CHEAP_SALARY) & (pool["value"] < VALUE_FLOOR)]
    swaps, unfixed = [], []
    for i in low:
        picked = pool["required_group"].notna()
        in_game = pool.loc[picked, "team"].map(game_of).value_counts().to_dict()
        open_slots = roster_size - int(picked.sum())
        budget = salary_cap - pool.loc[picked, "salary"].sum() + pool.loc[i, "salary"] - open_slots * cheapest_flex
        game = game_of.get(pool.loc[i, "team"])
        options = pool[(pool["position"] == pool.loc[i, "position"]) & ~picked
                       & (pool["salary"] <= budget) & (pool["value"] > pool.loc[i, "value"])]
        options = options[has_role(options)]
        options = options[[game_of.get(t) == game or in_game.get(game_of.get(t), 0) < MAX_FROM_GAME
                           for t in options["team"]]]
        if options.empty:
            unfixed.append(i)
            continue
        same_game = options[options["team"].map(game_of) == game]
        clearing = [s[s["value"] >= VALUE_FLOOR] for s in (same_game, options)]
        clearing = next((c for c in clearing if not c.empty), None)
        if clearing is not None:
            j = clearing.sort_values(["projected_points", "value"]).index[-1]
        else:
            j = options.sort_values(["value", "projected_points"]).index[-1]
        label = pool.loc[i, "required_group"]
        pool.loc[j, "required_group"], pool.loc[i, "required_group"] = label, None
        swaps.append(f"{label} {pool.loc[i, 'name']} ({pool.loc[i, 'value']:.1f}) -> "
                     f"{pool.loc[j, 'name']} ({pool.loc[j, 'team']}, ${pool.loc[j, 'salary']:,}, "
                     f"{pool.loc[j, 'value']:.1f})")
    if swaps:
        notes.append(f"Value check, rule picks under ${CHEAP_SALARY:,} below {VALUE_FLOOR:g} pts per $1K "
                     "swapped for better values (pts per $1K): " + "; ".join(swaps) + ".")
    if unfixed:
        notes.append("Value check, no better value with a real role fits for: " + ", ".join(
            f"{pool.loc[i, 'name']} ({pool.loc[i, 'value']:.1f})" for i in unfixed) + "; kept.")

    tagged = pool["required_group"].notna()
    in_game = pool.loc[tagged, "team"].map(game_of).value_counts().to_dict()

    def ranked(team: str, position: str) -> pd.DataFrame:
        on_team = pool[(pool["team"] == team) & (pool["position"] == position)]
        by = ["projected_points", "salary"] if position == "RB" else ["target_share", "projected_points"]
        return on_team.sort_values(by, ascending=False)

    qb = pool[pool["required_group"] == "QB"]
    qb_team = qb["team"].iloc[0] if not qb.empty else None
    qb_opp = qb["opponent"].iloc[0] if not qb.empty else None
    later_teams = [t for g in games.iloc[1:].itertuples() for t in (g.home, g.away)]

    candidates: dict[int, str] = {}  # row index -> why it's a candidate

    def add(rows: pd.Index, why: str) -> None:
        for i in rows:
            candidates.setdefault(i, why)

    for team in later_teams:
        add(ranked(team, "WR").index[1:2], "WR2, second or third game")
        add(ranked(team, "RB").index[:2], "RB, second or third game")
        add(ranked(team, "TE").index[:1], "top TE, second or third game")
    # Update 5: more stack and bring-back pieces from the top game.
    if qb_team is not None:
        def untagged_wrs(team: str) -> list[int]:
            return [i for i in ranked(team, "WR").index if pd.isna(pool.loc[i, "required_group"])]

        add(untagged_wrs(qb_team)[:1], "stack, the QB's next WR")
        add(ranked(qb_team, "TE").index[:1], "stack, the QB's TE")
        add(untagged_wrs(qb_opp)[:1], "second bring-back")
    # Update 1: whoever inherits an injured leader's role.
    fill_ins = set()
    for position, col in (("RB", "carries"), ("WR", "target_share"), ("TE", "target_share")):
        for team in _leader_out_teams(pool, stats, set(implied), position, col):
            on_team = ranked(team, position)
            if not on_team.empty:
                fill_ins.add(on_team.index[0])
                add(on_team.index[:1], f"fills in for {team}'s injured {position}1")

    left = salary_cap - pool.loc[tagged, "salary"].sum()
    kept, dropped = [], {"role": [], "game script": [], "salary or game cap": []}
    for i, why in candidates.items():
        row = pool.loc[i]
        team, opp = row["team"], row["opponent"]
        if pd.notna(row["required_group"]) or row["position"] not in FLEX_ELIGIBLE:
            continue
        if row["salary"] > left or in_game.get(game_of.get(team), 0) >= MAX_FROM_GAME:
            dropped["salary or game cap"].append(row["name"])
            continue
        # Update 2: a real role, unless he's inheriting one.
        if i not in fill_ins and (
            row["carry_share"] < FLEX_MIN_CARRY_SHARE if row["position"] == "RB"
            else row["target_share"] < FLEX_MIN_TARGET_SHARE
        ):
            dropped["role"].append(row["name"])
            continue
        # Update 4: RBs on teams that lead, pass catchers on teams that throw.
        edge = implied.get(team, 0) - implied.get(opp, 0)
        if (row["position"] == "RB" and edge <= 0 and implied.get(team, 0) < FLEX_RB_MIN_IMPLIED) or (
            row["position"] != "RB" and edge >= BIG_FAVORITE
        ):
            dropped["game script"].append(row["name"])
            continue
        kept.append(i)

    for reason, names in dropped.items():
        if names:
            notes.append(f"FLEX passes over ({reason}): {', '.join(names)}.")

    # Update 6: value floor, relaxed to the best values when too few clear it.
    flex = [i for i in kept if pool.loc[i, "value"] >= VALUE_FLOOR]
    if len(flex) < 2 and kept:
        flex = sorted(kept, key=lambda i: pool.loc[i, "value"], reverse=True)[:VALUE_FALLBACK_COUNT]
        notes.append(f"Fewer than two FLEX candidates project {VALUE_FLOOR:g}+ pts per $1K; "
                     f"keeping the {len(flex)} best values.")
    if flex:
        pool.loc[flex, "required_group"] = "FLEX"
        notes.append(f"FLEX, one of {len(flex)} at ${left:,.0f} or less: " + ", ".join(
            f"{pool.loc[i, 'name']} ({pool.loc[i, 'team']} {pool.loc[i, 'position']}, "
            f"{pool.loc[i, 'value']:.1f} pts/$1K, {candidates[i]})"
            for i in sorted(flex, key=lambda i: pool.loc[i, "value"], reverse=True)
        ) + ".")
    else:
        notes.append("No FLEX candidate clears the Echo Rule; FLEX is the optimizer's choice.")

    pool = pool.drop(columns=["match_key", "target_share", "carries", "carry_share", "value"])
    return pool, notes


def build_echo_pool(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fetches this week's lines and stats, then applies apply_echo_rule."""
    week_info = build_eligibility()
    season, week = week_info["season"], week_info["week"]
    context = _load_week_context(season, week)
    stats = get_player_season_stats(season)
    extra = []
    try:
        defense_vs = get_defense_vs_receivers(season)
    except Exception as exc:
        defense_vs = None
        extra.append(f"Defense-vs-receiver history unavailable ({exc}); WR3 and TE treat every "
                     "defense as average.")
    pool, notes = apply_echo_rule(
        pool, context["odds"], stats, context["rush_yds_allowed"], context["pass_yds_allowed"],
        defense_vs,
    )
    notes[:0] = [f"Week {week} lines from ESPN (DraftKings)."] + extra
    return pool, notes
