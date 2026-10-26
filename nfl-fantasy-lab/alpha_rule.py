"""The "Alpha Rule" lineup strategy: works only from the three games on the
uploaded slate with the highest Vegas over/under among those kicking off
Sunday in the 1pm and 4pm ET windows, and locks eight of the nine slots by
these rules, in this order:

  QB:   the starting QB of the team implied to score more in the game with
        the highest Vegas over/under.
  WR1:  that QB's WR1 (his starting receiver) -- the stack.
  WR2:  the WR1 of the team he's playing -- the "bring-back".
  WR swap: in the WR1 and WR2 slots, if the team's WR1 faces a top-10 pass
        defense (fewest pass yards allowed/game, league-wide), the team's
        WR2 takes the slot instead.
  Two-player bring-back: when the QB's WR1 draws few targets (under a 22%
        target share), the stack is dropped and the bring-back takes the
        opposing WR1 (WR2 slot) plus the opposing TE (TE slot, by target
        share); the optimizer fills the open WR slot.
  RB1:  the fill-in for an injured lead back. With none on the slate, RB1 is
        picked like RB2.
  RB2:  a lead back on a team that's favored or implied for 24+, from a
        team without an RB in the lineup yet. One RB slot may instead go to
        an underdog's workhorse: a lead back with most (over 50%) of his
        team's carries this season. Ranked: on a team starting a backup QB
        (it leans on the run) or facing a team whose WR1 is out, then the
        weaker rush defense (most rush yards allowed/game), then the better
        projection.
  WR3:  from the second- or third-highest over/under game, never a team
        starting a backup QB. On each team, the WR1 or WR2 -- whichever role
        the opposing defense has given up more to, against what those
        receivers score elsewhere -- with an 18%+ target share; the
        highest-implied team gets the slot.
  TE:   a TE on a team implied for 23+, with targets inside the 10 this
        season. Ranked by target share, then the FanDuel points the opponent
        allows TEs per game. Skipped when the two-player bring-back took
        the TE.
  DEF:  the defense facing the worst offense in these games (lowest
        Vegas-implied score); ties go to the better-projected defense. Never
        one facing any QB, RB, WR or TE the rules picked; the players it
        faces are then dropped from the pool so FLEX can't face it either.
  FLEX: a WR2, RB or TE from the second- or third-highest over/under
        game -- on each of those teams, its WR2, its two best-projected RBs
        (lead back included) and its top TE by target share -- whichever of
        them the optimizer likes best.

No more than four lineup players come from any one game; the rules skip a
candidate whose game is full, and the optimizer enforces it for FLEX (see
optimizer.py).

The rules also stay under the salary cap: each RB, WR3, TE and DEF pick
must leave enough to fill every slot still open at its cheapest salary on
the slate (for FLEX, the cheapest FLEX candidate), or the rule moves to its
next-ranked candidate (and the notes say so). Picks are made in the order above, so a tight cap squeezes the
later ones first.

A team's "point total" is its Vegas-implied score (see
vegas_strategy.implied_points). A team's WR1 is its WR on the slate with the
highest target share, its WR2 the next, its lead back its best-projected RB.
A team's QB1 is out when it's starting a backup (see vegas_strategy.py). Its
WR1 is out when its season leader in target share is off the (injury-
filtered) slate -- its top WR left then counts as WR1; its lead back is
injured when its season leader in carries is, and the fill-in is its
best-projected RB left on the slate. Where more than one fill-in is
available, the one on a team starting a backup QB or facing a team whose
WR1 is out comes first, then the weaker rush defense.

How a defense treats each WR role (see defense_vs_receivers): for every
game it played, each opponent's WR1 and WR2 (by season target share) score
FanDuel points (half PPR) against it, minus what that receiver averages in
his other games; the defense's number for a role is the average of those.

Picks are handed to the optimizer via the "required_group" column, same as
the Top-3 Vegas games strategy. When a rule can't be met it's skipped or
relaxed, and the notes shown on the page say so.
"""
import json
import os
import time

import pandas as pd

from matchup_filters import CACHE_DIR, STATS_PLAYER_WEEK_URL, build_eligibility
from optimizer import DEFAULT_SALARY_CAP, FLEX_ELIGIBLE
from real_stats import normalize_name
from vegas_strategy import (
    STATS_CACHE_MAX_AGE_SECONDS,
    _load_week_context,
    _starting_qbs,
    get_player_season_stats,
    implied_points,
    top_games,
)

KICKOFF_TIMEZONE = "America/New_York"
KICKOFF_HOURS = (13, 16)  # the 1:00 and 4:05/4:25 ET windows
SUNDAY = 6  # pandas dayofweek
ALPHA_GAME_COUNT = 3
TOP_N_PASS_DEFENSE = 10  # a "good" pass D, for the WR swap
RB_MIN_IMPLIED = 24.0
WORKHORSE_CARRY_SHARE = 0.5  # "most of his team's carries", for an underdog's back
LOW_STACK_TARGET_SHARE = 0.22  # QB's WR1 below this: two-player bring-back instead
WR3_MIN_TARGET_SHARE = 0.18
TE_MIN_IMPLIED = 23.0
MAX_FROM_GAME = 4
OFFENSE = ("QB", "RB", "WR", "TE")


def afternoon_games(odds: pd.DataFrame) -> pd.DataFrame:
    """The games kicking off Sunday in the 1pm and 4pm ET windows."""
    kickoff = pd.to_datetime(odds["kickoff"], utc=True).dt.tz_convert(KICKOFF_TIMEZONE)
    return odds[kickoff.dt.hour.isin(KICKOFF_HOURS) & (kickoff.dt.dayofweek == SUNDAY)]


def defense_vs_receivers(weekly: pd.DataFrame) -> dict:
    """From nflverse weekly player stats, how each defense treats pass
    catchers (see the module docstring). Returns
    {"wr_role": {defense: {"WR1": pts, "WR2": pts}}, "te": {defense: pts}}:
    the WR numbers are FanDuel points above (+) or below (-) what those
    receivers score elsewhere, the TE number is FanDuel points allowed to
    TEs per game."""
    weekly = weekly[weekly["season_type"] == "REG"]
    weekly = weekly.assign(fd=weekly["fantasy_points"] + 0.5 * weekly["receptions"])

    wr = weekly[weekly["position"] == "WR"]
    share = wr.groupby(["team", "player_id"])["target_share"].mean().reset_index()
    share["role"] = share.groupby("team")["target_share"].rank(ascending=False, method="first")
    wr = wr.merge(share.loc[share["role"] <= 2, ["team", "player_id", "role"]], on=["team", "player_id"])
    totals = wr.groupby(["team", "player_id"])["fd"].agg(fd_sum="sum", games="count")
    wr = wr.join(totals, on=["team", "player_id"])
    wr = wr[wr["games"] > 1]
    wr = wr.assign(delta=wr["fd"] - (wr["fd_sum"] - wr["fd"]) / (wr["games"] - 1))
    wr_role: dict[str, dict[str, float]] = {}
    for (defense, role), delta in wr.groupby(["opponent_team", "role"])["delta"].mean().items():
        wr_role.setdefault(defense, {})[f"WR{int(role)}"] = round(float(delta), 2)

    te = weekly[weekly["position"] == "TE"]
    te_allowed = te.groupby(["opponent_team", "week"])["fd"].sum().groupby("opponent_team").mean()
    return {"wr_role": wr_role, "te": {d: round(float(p), 2) for d, p in te_allowed.items()}}


def get_defense_vs_receivers(season: int, force_refresh: bool = False) -> dict:
    """defense_vs_receivers for `season`, cached like the season stats."""
    cache_file = f"{CACHE_DIR}/alpha_defense_vs_{season}.json"
    if not force_refresh and os.path.exists(cache_file):
        if time.time() - os.path.getmtime(cache_file) < STATS_CACHE_MAX_AGE_SECONDS:
            with open(cache_file) as f:
                return json.load(f)
    result = defense_vs_receivers(pd.read_parquet(STATS_PLAYER_WEEK_URL.format(season)))
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result, f)
    return result


def _leader_out_teams(pool: pd.DataFrame, stats: pd.DataFrame, teams, position: str, col: str) -> set[str]:
    """Teams whose season leader in `col` at `position` is off the slate."""
    out = set()
    for team in teams:
        leaders = stats[(stats["team"] == team) & (stats["position"] == position) & (stats[col] > 0)]
        if leaders.empty:
            continue
        leader_key = leaders.sort_values(col).iloc[-1]["match_key"]
        on_slate = pool.loc[(pool["team"] == team) & (pool["position"] == position), "match_key"]
        if leader_key not in set(on_slate):
            out.add(team)
    return out


def _worst(rank: int) -> str:
    """0 -> "worst", 1 -> "second-worst", ..."""
    if rank == 0:
        return "worst"
    return f"{('second', 'third')[rank - 1]}-worst" if rank < 3 else f"No. {rank + 1} worst"


def apply_alpha_rule(
    pool: pd.DataFrame,
    odds: pd.DataFrame,
    player_stats: pd.DataFrame,
    rush_yds_allowed: dict[str, float],
    pass_yds_allowed: dict[str, float],
    defense_vs: dict | None = None,
    salary_cap: float = DEFAULT_SALARY_CAP,
) -> tuple[pd.DataFrame, list[str]]:
    """Restricts `pool` to the top-3 Sunday 1pm and 4pm games and tags each
    rule's pick in a "required_group" column. Returns (pool, notes).

    player_stats columns: match_key, team, position, pass_attempts, carries,
    target_share, goal_line_targets (others are ignored). The yards-allowed
    dicts cover the whole league, so a "top-10 pass D" is league-wide.
    `defense_vs` is defense_vs_receivers' output; without it every defense
    counts as average against WRs and TEs.
    """
    defense_vs = defense_vs or {}
    wr_role_allowed = defense_vs.get("wr_role", {})
    te_allowed = defense_vs.get("te", {})

    notes: list[str] = []
    games = top_games(afternoon_games(odds), set(pool["team"]), n=ALPHA_GAME_COUNT)
    if games.empty:
        raise ValueError("No Sunday 1pm or 4pm games with Vegas lines match the uploaded slate.")

    opponent, implied, game_of = {}, {}, {}
    for _, g in games.iterrows():
        opponent[g["home"]], opponent[g["away"]] = g["away"], g["home"]
        game_of[g["home"]] = game_of[g["away"]] = f"{g['away']} @ {g['home']}"
        implied.update(implied_points(g))
    notes.append(f"Top {ALPHA_GAME_COUNT} Sunday 1pm and 4pm games by over/under: " + ", ".join(
        f"{g.away} @ {g.home} ({g.total:g})" for g in games.itertuples()
    ))

    def favorite(game: pd.Series) -> str:
        # The team implied to score more; a pick'em goes to the home team.
        return max((game["home"], game["away"]), key=implied.get)

    # Teams in the second and third games: where WR3 and FLEX come from.
    later_teams = [t for g in games.iloc[1:].itertuples() for t in (g.home, g.away)]

    pool = pool[pool["team"].isin(opponent)].copy()
    pool["opponent"] = pool["team"].map(opponent)
    pool["required_group"] = None

    stats = player_stats.drop_duplicates(["match_key", "team", "position"])
    stat_cols = ["pass_attempts", "carries", "target_share", "goal_line_targets"]
    pool["match_key"] = pool["name"].apply(normalize_name)
    pool = pool.merge(
        stats[["match_key", "team", "position"] + stat_cols],
        on=["match_key", "team", "position"], how="left",
    )
    for col in stat_cols:
        pool[col] = pool[col].fillna(0)

    starters, backup_teams = _starting_qbs(pool, stats, opponent)
    if backup_teams:
        notes.append("QB1 out, starting a backup: " + ", ".join(
            f"{t} ({pool.loc[starters[t], 'name']})" for t in sorted(backup_teams)
        ))
    wr1_out_teams = _leader_out_teams(pool, stats, opponent, "WR", "target_share")
    if wr1_out_teams:
        notes.append("WR1 out: " + ", ".join(sorted(wr1_out_teams)) + ".")
    rb_out_teams = _leader_out_teams(pool, stats, opponent, "RB", "carries")
    if rb_out_teams:
        notes.append("Lead back out: " + ", ".join(sorted(rb_out_teams)) + ".")

    def tag(i: int | list[int], label: str) -> None:
        pool.loc[i, "required_group"] = label

    def open_spot(team: str) -> bool:
        # Room under the per-game cap for one more rule pick from `team`'s game.
        tagged = pool.loc[pool["required_group"].notna(), "team"].map(game_of)
        return int((tagged == game_of[team]).sum()) < MAX_FROM_GAME

    def untagged(i: int) -> bool:
        return pd.isna(pool.loc[i, "required_group"])

    cheapest = pool.groupby("position")["salary"].min().to_dict()
    cheapest["ANY_FLEX"] = min(cheapest.get(p, 0) for p in FLEX_ELIGIBLE)

    def flex_candidates() -> list[int]:
        # Each second- or third-game team's WR2, two best-projected RBs
        # (lead back included) and top TE not already picked by a rule.
        out = []
        for team in later_teams:
            on_team = pool[pool["team"] == team]
            wrs = on_team[on_team["position"] == "WR"].sort_values(
                ["target_share", "projected_points"], ascending=False)
            rbs = on_team[on_team["position"] == "RB"].sort_values(
                ["projected_points", "salary"], ascending=False)
            tes = on_team[on_team["position"] == "TE"].sort_values(
                ["target_share", "projected_points"], ascending=False)
            out += list(wrs.index[1:2]) + list(rbs.index[:2]) + list(tes.index[:1])
        return [i for i in out if untagged(i)]

    def def_reserve(picked_teams: set[str]) -> float:
        # The cheapest DEF the DEF rule could still take: one facing none of
        # `picked_teams`, or failing that, one not facing the QB.
        defs = pool[pool["position"] == "DEF"]
        free = defs[~defs["opponent"].isin(picked_teams)]
        return (free if not free.empty else defs[defs["opponent"] != qb_team])["salary"].min()

    def affordable(ranked: list[int], label: str, still_open: list[str]) -> int | None:
        # The best-ranked candidate that leaves enough under the cap to fill
        # `still_open` at the cheapest salaries on the slate (for the DEF, the
        # cheapest one that would still be eligible).
        picked = pool["required_group"].notna()
        spent = pool.loc[picked, "salary"].sum()
        picked_teams = set(pool.loc[picked & pool["position"].isin(OFFENSE), "team"])
        flex_pool = [i for i in flex_candidates() if i not in ranked]
        cheapest["FLEX"] = pool.loc[flex_pool, "salary"].min() if flex_pool else cheapest["ANY_FLEX"]
        reserve = sum(cheapest.get(p, 0) for p in still_open if p != "DEF")
        for n, i in enumerate(ranked):
            own = def_reserve(picked_teams | {pool.loc[i, "team"]}) if "DEF" in still_open else 0
            if spent + pool.loc[i, "salary"] + reserve + own <= salary_cap:
                if n:
                    notes.append(f"{label}: {', '.join(pool.loc[ranked[:n], 'name'])} would leave too "
                                 f"little of the ${salary_cap:,.0f} cap for the open slots; next best.")
                return i
        notes.append(f"No {label} candidate fits under the ${salary_cap:,.0f} cap; {label} rule skipped.")
        return None

    # QB
    qb_team = favorite(games.iloc[0])
    if qb_team not in starters:
        notes.append(f"No {qb_team} QB on the slate; the QB comes from {opponent[qb_team]} instead.")
        qb_team = opponent[qb_team]
    if qb_team not in starters:
        raise ValueError("no QB on the slate in the highest over/under game")
    qb, qb_opp = starters[qb_team], opponent[qb_team]
    tag(qb, "QB")
    notes.append(
        f"QB {pool.loc[qb, 'name']} ({qb_team} vs {qb_opp}): implied {implied[qb_team]:.1f} pts "
        "in the highest over/under game."
    )

    def team_wrs(team: str) -> pd.DataFrame:
        return pool[(pool["position"] == "WR") & (pool["team"] == team)].sort_values(
            ["target_share", "projected_points"], ascending=False
        )

    # WR1 stack and WR2 bring-back; either swaps to the team's WR2 when its
    # WR1 faces a top-10 pass D.
    good_pass_d = set(sorted(pass_yds_allowed, key=pass_yds_allowed.get)[:TOP_N_PASS_DEFENSE])

    def tag_wr1(team: str, label: str, why: str) -> None:
        ranked = team_wrs(team)
        if ranked.empty:
            notes.append(f"No {team} WR on the slate; {label} rule skipped.")
            return
        i, opp = ranked.index[0], opponent[team]
        if opp in good_pass_d:
            faces = (f"{pool.loc[i, 'name']} faces {opp}, a top-{TOP_N_PASS_DEFENSE} pass D "
                     f"({pass_yds_allowed[opp]:.0f} pass yds allowed/game)")
            if len(ranked) > 1:
                i = ranked.index[1]
                why += f" -- swapped to {team}'s WR2 since WR1 {faces}"
            else:
                why += f"; WR1 {faces}, but {team} has no WR2 on the slate"
        tag(i, label)
        notes.append(f"{label} {pool.loc[i, 'name']}: {why} "
                     f"({pool.loc[i, 'target_share']:.0%} target share).")

    # Two-player bring-back: when the QB's WR1 draws few targets, the stack
    # gives way to the opposing WR1 plus the opposing TE.
    stack_wrs = team_wrs(qb_team)
    opp_tes = pool[(pool["position"] == "TE") & (pool["team"] == qb_opp)].sort_values(
        ["target_share", "projected_points"], ascending=False)
    stack_share = stack_wrs["target_share"].iloc[0] if not stack_wrs.empty else 0.0
    double_bring_back = (not stack_wrs.empty and stack_share < LOW_STACK_TARGET_SHARE
                         and not opp_tes.empty)
    if double_bring_back:
        notes.append(f"Two-player bring-back: {qb_team}'s WR1 {stack_wrs.iloc[0]['name']} has only a "
                     f"{stack_share:.0%} target share (under {LOW_STACK_TARGET_SHARE:.0%}), so no stack; "
                     f"the bring-back takes {qb_opp}'s WR1 and TE, and the optimizer fills the open WR.")
    else:
        tag_wr1(qb_team, "WR1", "the QB's WR1")
    tag_wr1(qb_opp, "WR2", f"WR1 of the opposing team ({qb_opp})")
    if double_bring_back:
        t = opp_tes.index[0]
        tag(t, "TE")
        notes.append(f"TE {pool.loc[t, 'name']}: the second half of the bring-back, {qb_opp}'s top TE "
                     f"({pool.loc[t, 'target_share']:.0%} target share).")

    # RB1: the fill-in for an injured lead back. RB2 (and RB1 when there's no
    # fill-in): a lead back on a favored or 24+ team, ranked by matchup.
    rb = pool["position"] == "RB"
    lead_backs = {
        team: backs.sort_values(["projected_points", "salary"]).index[-1]
        for team, backs in pool[rb].groupby("team")
    }
    worst_rush = sorted((t for t in opponent if t in rush_yds_allowed),
                        key=rush_yds_allowed.get, reverse=True)
    def matchup(team: str, faced: str) -> str:
        rush = (f"{faced}, the {_worst(worst_rush.index(faced))} rush D "
                f"({rush_yds_allowed[faced]:.0f} rush yds allowed/game)") if faced in worst_rush else faced
        out = f"faces {rush}" + (", whose WR1 is out" if faced in wr1_out_teams else "")
        if team in backup_teams:
            out += f"; {team} is starting a backup QB, so it leans on the run"
        return out

    def rb_rank(i: int) -> tuple:
        # A back whose own QB1 is out gets the work; so does one facing a
        # team down its WR1.
        team, faced = pool.loc[i, "team"], pool.loc[i, "opponent"]
        return (team in backup_teams or faced in wr1_out_teams, rush_yds_allowed.get(faced, 0),
                pool.loc[i, "projected_points"], pool.loc[i, "salary"])

    # Slots still open after each pick, for the salary check. The two-player
    # bring-back leaves a WR slot for the optimizer and has filled the TE.
    after = {"RB1": ["RB", "WR", "TE", "DEF", "FLEX"], "RB2": ["WR", "TE", "DEF", "FLEX"],
             "WR3": ["TE", "DEF", "FLEX"], "TE": ["DEF", "FLEX"], "DEF": ["FLEX"]}
    if double_bring_back:
        after = {k: [s for s in v if s != "TE"] + ["WR"] for k, v in after.items()}

    # Workhorses: lead backs with most of their team's carries this season.
    team_carries = stats.groupby("team")["carries"].sum()

    def carry_share(i: int) -> float:
        total = team_carries.get(pool.loc[i, "team"], 0)
        return pool.loc[i, "carries"] / total if total else 0.0

    rb_slots = ["RB1", "RB2"]
    fill_ins = [lead_backs[t] for t in sorted(rb_out_teams) if t in lead_backs and open_spot(t)]
    i = affordable(sorted(fill_ins, key=rb_rank, reverse=True), "RB1", after["RB1"]) if fill_ins else None
    if i is not None:
        tag(i, "RB1")
        team, faced = pool.loc[i, "team"], pool.loc[i, "opponent"]
        notes.append(f"RB1 {pool.loc[i, 'name']} ({team} vs {faced}): fills in for {team}'s injured "
                     f"lead back; {matchup(team, faced)}.")
        rb_slots = ["RB2"]
    else:
        notes.append("No injured lead back has a fill-in on the slate; RB1 is picked like RB2.")

    underdog_used = False  # only one RB slot may go to an underdog's workhorse
    for label in rb_slots:
        rb_teams = set(pool.loc[pool["required_group"].isin(["RB1", "RB2"]), "team"])
        candidates = [i for t, i in lead_backs.items() if t not in rb_teams and untagged(i) and open_spot(t)]
        favored = [i for i in candidates if implied[pool.loc[i, "team"]] >= RB_MIN_IMPLIED
                   or implied[pool.loc[i, "team"]] > implied[pool.loc[i, "opponent"]]]
        workhorses = [] if underdog_used else [
            i for i in candidates if i not in favored and carry_share(i) > WORKHORSE_CARRY_SHARE]
        if not candidates:
            notes.append(f"No lead back fits the {label} rule; {label} rule skipped.")
            continue
        if not favored and not workhorses:
            notes.append(f"No lead back's team is favored or implied for {RB_MIN_IMPLIED:g}+; "
                         f"{label} considers every lead back.")
        i = affordable(sorted(favored + workhorses or candidates, key=rb_rank, reverse=True),
                       label, after[label])
        if i is None:
            continue
        tag(i, label)
        team, faced = pool.loc[i, "team"], pool.loc[i, "opponent"]
        workhorse = ""
        if i in workhorses:
            underdog_used = True
            workhorse = (f"; an underdog's workhorse with {carry_share(i):.0%} of {team}'s carries "
                         "(the one RB slot allowed off a favored team)")
        notes.append(f"{label} {pool.loc[i, 'name']} ({team} vs {faced}, implied "
                     f"{implied[team]:.1f}): {matchup(team, faced)}{workhorse}.")

    # WR3: the role the defense handles worse, from the second or third game.
    if len(games) > 1:
        wr3_teams = later_teams
        skipped = sorted(t for t in wr3_teams if t in backup_teams)
        if skipped:
            notes.append(f"WR3 passes over {', '.join(skipped)}: starting a backup QB.")
        candidates = []  # (team, row index, role, points vs role)
        for team in wr3_teams:
            if team in backup_teams or not open_spot(team):
                continue
            ranked = team_wrs(team)
            allowed = wr_role_allowed.get(opponent[team], {})
            roles = [(i, f"WR{n + 1}", allowed.get(f"WR{n + 1}", 0.0))
                     for n, i in enumerate(ranked.index[:2])
                     if untagged(i) and pool.loc[i, "target_share"] >= WR3_MIN_TARGET_SHARE]
            if roles:
                i, role, pts = max(roles, key=lambda r: (r[2], r[1] == "WR1"))
                candidates.append((team, i, role, pts))
        candidates.sort(key=lambda c: (implied[c[0]], pool.loc[c[1], "projected_points"]), reverse=True)
        pick = affordable([c[1] for c in candidates], "WR3", after["WR3"]) if candidates else None
        if pick is not None:
            team, i, role, pts = next(c for c in candidates if c[1] == pick)
            opp = opponent[team]
            other = "WR2" if role == "WR1" else "WR1"
            vs = wr_role_allowed.get(opp, {})
            why = (f"{opp} allows {vs[role]:+.1f} FD pts/game to {role}s vs what they score elsewhere "
                   f"({vs.get(other, 0.0):+.1f} to {other}s)" if role in vs
                   else f"no {opp} history against WRs, so the WR1")
            tag(i, "WR3")
            notes.append(f"WR3 {pool.loc[i, 'name']} ({team} {role} vs {opp}, implied {implied[team]:.1f}): "
                         f"{why} ({pool.loc[i, 'target_share']:.0%} target share).")
        elif not candidates:
            notes.append(f"No WR in the second or third game has a {WR3_MIN_TARGET_SHARE:.0%}+ target share "
                         "on a team with its starting QB; WR3 rule skipped.")
    else:
        notes.append("Only one Sunday 1pm or 4pm game on the slate; WR3 rule skipped.")

    # TE: red-zone targets on a 23+ team, ranked by target share then the
    # points the opponent allows TEs.
    te = pool[(pool["position"] == "TE") & pool["team"].map(open_spot)]
    high = te[te["team"].map(implied) >= TE_MIN_IMPLIED]
    goal_line = high[high["goal_line_targets"] > 0]
    te_candidates = goal_line if not goal_line.empty else high if not high.empty else te
    if goal_line.empty and not double_bring_back:
        notes.append("No TE on a 23+ team has a target inside the 10; TE rule relaxed"
                     + ("." if not high.empty else " to every TE."))
    t = None
    if double_bring_back:
        pass  # the bring-back took the TE
    elif te_candidates.empty:
        notes.append("No TE fits under the per-game cap; TE rule skipped.")
    else:
        te_candidates = te_candidates.assign(opp_te=te_candidates["opponent"].map(te_allowed).fillna(0))
        ranked = te_candidates.sort_values(["target_share", "opp_te", "projected_points"], ascending=False)
        t = affordable(list(ranked.index), "TE", after["TE"])
    if t is not None:
        tag(t, "TE")
        opp = pool.loc[t, "opponent"]
        allows = f", {opp} allows {te_allowed[opp]:.1f} FD pts/game to TEs" if opp in te_allowed else ""
        notes.append(f"TE {pool.loc[t, 'name']} ({pool.loc[t, 'team']} vs {opp}, implied "
                     f"{implied[pool.loc[t, 'team']]:.1f}): {pool.loc[t, 'target_share']:.0%} target share, "
                     f"{pool.loc[t, 'goal_line_targets']:.0f} targets inside the 10{allows}.")

    # DEF: facing the worst offense, never any offense the rules picked from.
    picked_offenses = set(pool.loc[pool["required_group"].notna() & pool["position"].isin(OFFENSE), "team"])
    defs = pool[(pool["position"] == "DEF") & ~pool["opponent"].isin(picked_offenses)].copy()
    if defs.empty:
        notes.append("Every defense faces a player the rules picked; DEF only avoids the QB's team.")
        picked_offenses = {qb_team}
        defs = pool[(pool["position"] == "DEF") & (pool["opponent"] != qb_team)].copy()
    d = None
    if defs.empty:
        notes.append("No defense is free of the lineup's QB; DEF rule skipped.")
    else:
        defs["neg_implied"] = -defs["opponent"].map(implied)
        d = affordable(list(defs.sort_values(["neg_implied", "projected_points"], ascending=False).index),
                       "DEF", after["DEF"])
    if d is not None:
        tag(d, "DEF")
        faced = pool.loc[d, "opponent"]
        worst = min(implied, key=implied.get)
        scope = "worst offense" if faced == worst else (
            f"worst offense not already in the lineup ({worst} is)"
        )
        notes.append(f"DEF {pool.loc[d, 'name']}: faces {faced}, the {scope}, "
                     f"implied {implied[faced]:.1f} pts.")
        # FLEX (or anyone else the optimizer adds) can't face the DEF either.
        facing = pool["position"].isin(OFFENSE) & (pool["team"] == faced) & pool["required_group"].isna()
        pool = pool[~facing]

    # FLEX: a WR2, RB or TE from the second or third game, in a game with
    # room left and at a salary that still fits; the optimizer picks among them.
    left = salary_cap - pool.loc[pool["required_group"].notna(), "salary"].sum()
    flex = [i for i in flex_candidates()
            if open_spot(pool.loc[i, "team"]) and pool.loc[i, "salary"] <= left]
    if flex:
        tag(flex, "FLEX")
        which = "the only WR2, RB or TE" if len(flex) == 1 else f"one of {len(flex)} WR2s, RBs and TEs"
        notes.append(f"FLEX, {which} from the second and third games "
                     f"at ${left:,.0f} or less: " + ", ".join(
                         f"{pool.loc[i, 'name']} ({pool.loc[i, 'team']} {pool.loc[i, 'position']})"
                         for i in pool.loc[flex].sort_values("projected_points", ascending=False).index
                     ) + ".")
    elif later_teams:
        notes.append(f"No WR2, RB or TE from the second or third game fits under the cap and the "
                     f"per-game limit; FLEX is the optimizer's choice.")
    else:
        notes.append("Only one game; FLEX is the optimizer's choice.")

    pool["max_from_game"] = MAX_FROM_GAME
    pool = pool.drop(columns=["match_key"] + stat_cols)
    return pool.reset_index(drop=True), notes


def build_alpha_pool(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fetches this week's lines and stats, then applies apply_alpha_rule."""
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
    pool, notes = apply_alpha_rule(
        pool, context["odds"], stats, context["rush_yds_allowed"], context["pass_yds_allowed"],
        defense_vs,
    )
    notes[:0] = [f"Week {week} lines from ESPN (DraftKings)."] + extra
    return pool, notes
