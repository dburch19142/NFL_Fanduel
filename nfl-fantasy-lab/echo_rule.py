"""The "Echo Rule" lineup strategy: the Alpha Rule (see alpha_rule.py) and
the Beta strategy's running backs (see beta_strategy.py), with seven updates
aimed at the cheap picks -- the players who have to outscore their salary
for a lineup to win. The Alpha Rule picks QB, WR1, WR2, WR3, TE and DEF --
including its two-player bring-back (the opposing WR1 and TE in place of
the stack when the QB's WR1 draws under a 22% target share) and its
underdog workhorse back; Beta, when it can, the RBs; the updates change the
games it works from, the backs it can take, and how FLEX is filled.

  Beta RBs: a team with a top-10 rushing offense facing a bottom-10 rushing
        defense in a Sunday 1pm or 4pm game -- in any such game, not just
        the Alpha Rule's -- supplies the RBs. Their lead backs take RB1 and
        RB2 (one per team, the weaker rush defense first, then the better
        projection), as long as each fits under the cap with the rest of the
        lineup, keeps the per-game limit and isn't facing the lineup's DEF.
        With two Beta lead backs in, every other RB is dropped, so an RB at
        FLEX is a Beta back too. With one, he takes RB2 and the Alpha Rule's
        RB1 stays; with none, the Alpha Rule's RBs stay.

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
        -- that fits the cap and the per-game limit. When nothing fits, the
        most expensive other RB, WR or TE pick moves down to a cheaper
        player at its position with a real role (the QB's WR1 stays on the
        QB's team) to free salary, and the swap is retried -- taking the
        pairing that projects best, and only if the two moves together
        don't cost projected points. Every swap leaves room under the cap
        for a FLEX candidate that passes the Echo Rule (when one could at
        any salary), so FLEX isn't left to the optimizer. RB fill-ins for
        an injured lead back (rule 1), DEF and QB are left as picked.
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
from beta_strategy import load_rushing_ranks, qualifying_teams
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
    rushing_ranks: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """The Alpha Rule, Beta RBs and the seven updates in the module
    docstring. Arguments and return value match apply_alpha_rule's, plus
    `rushing_ranks` (beta_strategy.rushing_ranks' output); without it the
    Alpha Rule's RBs stay."""
    stats = player_stats.drop_duplicates(["match_key", "team", "position"])
    slate_teams = set(pool["team"])
    all_odds = odds
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
    slate_pool = pool

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

    # Beta: the qualifying teams, wherever they play on Sunday afternoon.
    beta = [q for q in qualifying_teams(rushing_ranks, all_odds) if q["team"] in slate_teams] \
        if rushing_ranks is not None else []
    beta_teams = {q["team"] for q in beta}
    beta_rank = {q["team"]: q["defense_rank"] for q in beta}
    for _, g in afternoon_games(all_odds).iterrows():
        if beta_teams & {g["home"], g["away"]}:
            implied.update(implied_points(g))
            game_of[g["home"]] = game_of[g["away"]] = f"{g['away']} @ {g['home']}"
    opponent_of = {}
    for _, g in afternoon_games(all_odds).iterrows():
        opponent_of[g["home"]], opponent_of[g["away"]] = g["away"], g["home"]
    on_pool = set(zip(pool["name"], pool["team"]))
    off_pool = pd.Series([k not in on_pool for k in zip(slate_pool["name"], slate_pool["team"])],
                         index=slate_pool.index, dtype=bool)
    # Not from the offense the DEF faces: the Alpha Rule dropped those.
    def_faces = set(pool.loc[pool["required_group"] == "DEF", "opponent"])
    extra = slate_pool[(slate_pool["position"] == "RB") & slate_pool["team"].isin(beta_teams - def_faces)
                       & off_pool]
    if not extra.empty:
        extra = extra.assign(opponent=extra["team"].map(opponent_of), required_group=None,
                             max_from_game=MAX_FROM_GAME)
        pool = pd.concat([pool, extra[[c for c in pool.columns if c in extra.columns]]], ignore_index=True)

    pool["required_group"] = pool["required_group"].where(pool["required_group"] != "FLEX")
    pool["match_key"] = pool["name"].apply(normalize_name)
    pool = pool.merge(stats[["match_key", "team", "position", "target_share", "carries"]],
                      on=["match_key", "team", "position"], how="left")
    pool = pool.merge(carry_shares(stats), on=["match_key", "team"], how="left")
    pool["target_share"] = pool["target_share"].fillna(0)
    pool["carry_share"] = pool["carry_share"].fillna(0)
    pool["value"] = pool.apply(value, axis=1)
    roster_size = sum(ROSTER_SLOTS.values()) + FLEX_COUNT
    cheapest_flex = pool.loc[pool["position"].isin(FLEX_ELIGIBLE), "salary"].min()

    # Beta RBs: their lead backs take the RB slots when they fit.
    if not beta:
        notes.append("Beta RBs: no team with a top-10 rushing offense faces a bottom-10 rushing defense "
                     "on Sunday afternoon" + ("" if rushing_ranks is not None else " (rushing ranks unavailable)")
                     + "; the Alpha Rule's RBs stay.")
    else:
        notes.append("Beta RB teams: " + ", ".join(
            f"{q['team']} vs {q['opponent']} (No. {q['offense_rank']} rush O vs No. {q['defense_rank']} "
            f"worst rush D)" for q in beta) + ".")
        rbs = pool[(pool["position"] == "RB") & pool["team"].isin(beta_teams)]
        leads = [backs.sort_values(["projected_points", "salary"]).index[-1] for _, backs in rbs.groupby("team")]
        leads.sort(key=lambda i: (beta_rank[pool.loc[i, "team"]], -pool.loc[i, "projected_points"]))
        alpha_rbs = [i for label in ("RB1", "RB2") for i in pool.index[pool["required_group"] == label]]
        def_faces = set(pool.loc[pool["required_group"] == "DEF", "opponent"])
        others = list(pool.index[pool["required_group"].notna() & ~pool.index.isin(alpha_rbs)])
        flex_reserve = (roster_size - len(others) - 2) * cheapest_flex
        all_rbs = pool.index[pool["position"] == "RB"]

        def fits(lineup: list[int], i: int, rb_reserve: float) -> bool:
            # i joins `lineup` under the cap (with `rb_reserve` held for an
            # RB slot still open), the per-game limit and away from the DEF.
            in_game = pool.loc[lineup, "team"].map(game_of).value_counts()
            return (pool.loc[lineup + [i], "salary"].sum() + rb_reserve + flex_reserve <= salary_cap
                    and pool.loc[i, "team"] not in def_faces
                    and in_game.get(game_of.get(pool.loc[i, "team"]), 0) < MAX_FROM_GAME)

        chosen = []
        for i in leads:
            if len(chosen) == 2:
                break
            if chosen and pool.loc[i, "team"] == pool.loc[chosen[0], "team"]:
                continue
            reserve = 0 if chosen else pool.loc[all_rbs.drop(i), "salary"].min()
            if fits(others + chosen, i, reserve):
                chosen.append(i)
        if len(chosen) == 1:
            keep = [r for r in alpha_rbs if pool.loc[r, "team"] != pool.loc[chosen[0], "team"]][:1]
            if not keep or not fits(others + keep, chosen[0], 0):
                chosen = []
        if len(chosen) == 2:
            pool.loc[alpha_rbs, "required_group"] = None
            pool.loc[chosen[0], "required_group"], pool.loc[chosen[1], "required_group"] = "RB1", "RB2"
            drop = (pool["position"] == "RB") & ~pool["team"].isin(beta_teams)
            pool = pool[~drop]
            notes.append("Beta RBs: " + " and ".join(
                f"{pool.loc[i, 'name']} ({pool.loc[i, 'team']} vs {pool.loc[i, 'opponent']}, "
                f"${pool.loc[i, 'salary']:,})" for i in chosen)
                + " take RB1 and RB2; every other RB is dropped.")
        elif chosen:
            pool.loc[[r for r in alpha_rbs if r not in keep], "required_group"] = None
            pool.loc[keep, "required_group"] = "RB1"
            pool.loc[chosen[0], "required_group"] = "RB2"
            notes.append(f"Beta RBs: only {pool.loc[chosen[0], 'name']} ({pool.loc[chosen[0], 'team']}) fits; "
                         "he takes RB2 and the Alpha Rule's RB1 stays.")
        else:
            notes.append("Beta RBs: no Beta lead back fits under the cap, the per-game limit and the DEF; "
                         "the Alpha Rule's RBs stay.")

    leader_out = [(position, team) for position, col in (("RB", "carries"), ("WR", "target_share"),
                                                         ("TE", "target_share"))
                  for team in _leader_out_teams(pool, stats, set(implied), position, col)]

    def has_role(rows: pd.DataFrame) -> pd.Series:
        # Update 2's bar: a 15%+ target share, or 40%+ of the carries for an RB.
        return rows["carry_share"].where(rows["position"] == "RB", rows["target_share"]) >= rows[
            "position"].map(lambda p: FLEX_MIN_CARRY_SHARE if p == "RB" else FLEX_MIN_TARGET_SHARE)

    def ranked(team: str, position: str) -> pd.DataFrame:
        on_team = pool[(pool["team"] == team) & (pool["position"] == position)]
        by = ["projected_points", "salary"] if position == "RB" else ["target_share", "projected_points"]
        return on_team.sort_values(by, ascending=False)

    qb = pool[pool["required_group"] == "QB"]
    qb_team = qb["team"].iloc[0] if not qb.empty else None
    qb_opp = qb["opponent"].iloc[0] if not qb.empty else None
    later_teams = [t for g in games.iloc[1:].itertuples() for t in (g.home, g.away)]

    def flex_pool(salary_check: bool = True) -> tuple[dict[int, str], list[int], dict[str, list[str]], float]:
        # FLEX candidates as the picks stand now (updates 1, 2, 4 and 5):
        # (row index -> why it's a candidate, the ones that pass, the names
        # passed over by reason, the salary left).
        tagged = pool["required_group"].notna()
        in_game = pool.loc[tagged, "team"].map(game_of).value_counts().to_dict()
        candidates: dict[int, str] = {}

        def add(rows, why: str) -> None:
            for i in rows:
                candidates.setdefault(i, why)

        for team in later_teams:
            add(ranked(team, "WR").index[1:2], "WR2, second or third game")
            add(ranked(team, "RB").index[:2], "RB, second or third game")
            add(ranked(team, "TE").index[:1], "top TE, second or third game")
        for team in sorted(beta_teams):
            add(ranked(team, "RB").index[:2], "RB, Beta rushing matchup")
        # Update 5: more stack and bring-back pieces from the top game.
        if qb_team is not None:
            def untagged_wrs(team: str) -> list[int]:
                return [i for i in ranked(team, "WR").index if pd.isna(pool.loc[i, "required_group"])]

            add(untagged_wrs(qb_team)[:1], "stack, the QB's next WR")
            add(ranked(qb_team, "TE").index[:1], "stack, the QB's TE")
            add(untagged_wrs(qb_opp)[:1], "second bring-back")
        # Update 1: whoever inherits an injured leader's role.
        fill_ins = set()
        for position, team in leader_out:
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
            if (salary_check and row["salary"] > left) or in_game.get(game_of.get(team), 0) >= MAX_FROM_GAME:
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
        return candidates, kept, dropped, left

    # Update 6, rule picks: a RB, WR or TE pick under $6,000 that projects
    # below the value floor gives way to a better value at its position with
    # a real role -- from its own game first, then any of these games; one
    # clearing the floor if there is one, else the best value -- as long as
    # it fits the cap and the per-game limit.
    # A back filling in for an injured lead back (update 1) is exempt: his
    # projection reflects his old role, not the carries he's inheriting.
    # Receivers aren't: an injured WR1's targets spread across the roster.
    rb_out = {team for position, team in leader_out if position == "RB"}
    filling_in = (pool["position"] == "RB") & pool["team"].isin(rb_out)
    low = pool.index[pool["required_group"].notna() & pool["position"].isin(FLEX_ELIGIBLE) & ~filling_in
                     & (pool["salary"] < CHEAP_SALARY) & (pool["value"] < VALUE_FLOOR)]
    def move(old: int, new: int) -> None:
        pool.loc[new, "required_group"], pool.loc[old, "required_group"] = pool.loc[old, "required_group"], None

    # When a FLEX candidate could pass the Echo Rule at some salary, every
    # swap must leave enough under the cap for one of them; otherwise the
    # swaps just hold back the cheapest FLEX-eligible salary.
    keep_flex_room = bool(flex_pool(salary_check=False)[1])

    def flex_fits() -> bool:
        return not keep_flex_room or bool(flex_pool()[1])

    def replacement(i: int) -> int | None:
        # The best better-value swap for pick i that fits the cap and game
        # limits as the picks stand now, and leaves room for a FLEX.
        picked = pool["required_group"].notna()
        in_game = pool.loc[picked, "team"].map(game_of).value_counts().to_dict()
        open_slots = roster_size - int(picked.sum()) - (1 if keep_flex_room else 0)
        budget = salary_cap - pool.loc[picked, "salary"].sum() + pool.loc[i, "salary"] - open_slots * cheapest_flex
        game = game_of.get(pool.loc[i, "team"])
        options = pool[(pool["position"] == pool.loc[i, "position"]) & ~picked
                       & (pool["salary"] <= budget) & (pool["value"] > pool.loc[i, "value"])]
        options = options[has_role(options)]
        if options.empty:
            return None
        options = options[[game_of.get(t) == game or in_game.get(game_of.get(t), 0) < MAX_FROM_GAME
                           for t in options["team"]]]
        # Preference: clearing the floor in its own game, then clearing it
        # anywhere (by projection), then the best value.
        same_game = options[options["team"].map(game_of) == game]
        order = []
        for group in (same_game[same_game["value"] >= VALUE_FLOOR], options[options["value"] >= VALUE_FLOOR]):
            order += list(group.sort_values(["projected_points", "value"], ascending=False).index)
        order += list(options.sort_values(["value", "projected_points"], ascending=False).index)
        for j in dict.fromkeys(order):
            move(i, j)
            fits = flex_fits()
            move(j, i)
            if fits:
                return j
        return None

    def downgrade_for(i: int) -> tuple[int, int, int] | None:
        # When nothing fits for pick i: free salary by moving the most
        # expensive other RB/WR/TE pick down to a cheaper player at its
        # position with a real role (the QB's WR1 stays on the QB's team),
        # then retry. Takes the pairing that projects best, and only one that
        # doesn't cost projected points overall. Returns (expensive pick,
        # its downgrade, pick i's replacement).
        picked = pool["required_group"].notna()
        others = pool[picked & pool["position"].isin(FLEX_ELIGIBLE) & ~filling_in & (pool.index != i)]
        if others.empty:
            return None
        e = others.sort_values(["salary", "projected_points"]).index[-1]
        game = game_of.get(pool.loc[e, "team"])
        in_game = pool.loc[picked, "team"].map(game_of).value_counts().to_dict()
        cheaper = pool[(pool["position"] == pool.loc[e, "position"]) & ~picked
                       & (pool["salary"] < pool.loc[e, "salary"])]
        cheaper = cheaper[has_role(cheaper)]
        if cheaper.empty:
            return None
        cheaper = cheaper[[game_of.get(t) == game or in_game.get(game_of.get(t), 0) < MAX_FROM_GAME
                           for t in cheaper["team"]]]
        if pool.loc[e, "required_group"] == "WR1":
            cheaper = cheaper[cheaper["team"] == pool.loc[e, "team"]]
        best, best_gain = None, 0.0
        for d in cheaper.index:
            move(e, d)
            r = replacement(i)
            move(d, e)
            if r is None:
                continue
            gain = (pool.loc[d, "projected_points"] - pool.loc[e, "projected_points"]
                    + pool.loc[r, "projected_points"] - pool.loc[i, "projected_points"])
            if gain >= best_gain:
                best, best_gain = (e, d, r), gain
        return best

    def describe(old: int, new: int) -> str:
        return (f"{pool.loc[new, 'required_group']} {pool.loc[old, 'name']} ({pool.loc[old, 'value']:.1f}) -> "
                f"{pool.loc[new, 'name']} ({pool.loc[new, 'team']}, ${pool.loc[new, 'salary']:,}, "
                f"{pool.loc[new, 'value']:.1f})")

    swaps, downgrades, unfixed = [], [], []
    for i in low:
        j = replacement(i)
        if j is None:
            plan = downgrade_for(i)
            if plan is None:
                unfixed.append(i)
                continue
            e, d, j = plan
            move(e, d)
            freed = pool.loc[e, "salary"] - pool.loc[d, "salary"]
            downgrades.append(f"{describe(e, d)}, freeing ${freed:,} for {pool.loc[i, 'name']}")
        move(i, j)
        swaps.append(describe(i, j))
    if downgrades:
        notes.append("Value check, most expensive pick downgraded to make room (pts per $1K): "
                     + "; ".join(downgrades) + ".")
    if swaps:
        notes.append(f"Value check, rule picks under ${CHEAP_SALARY:,} below {VALUE_FLOOR:g} pts per $1K "
                     "swapped for better values (pts per $1K): " + "; ".join(swaps) + ".")
    if unfixed:
        notes.append("Value check, no better value with a real role fits for: " + ", ".join(
            f"{pool.loc[i, 'name']} ({pool.loc[i, 'value']:.1f})" for i in unfixed)
            + ", even after downgrading the most expensive pick; kept.")

    candidates, kept, dropped, left = flex_pool()

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
    try:
        rushing = load_rushing_ranks(season)
    except Exception as exc:
        rushing = None
        extra.append(f"Rushing ranks unavailable ({exc}); no Beta RBs this week.")
    pool, notes = apply_echo_rule(
        pool, context["odds"], stats, context["rush_yds_allowed"], context["pass_yds_allowed"],
        defense_vs, rushing_ranks=rushing,
    )
    notes[:0] = [f"Week {week} lines from ESPN (DraftKings)."] + extra
    return pool, notes
