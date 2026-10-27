"""The "Charlie Rule" lineup strategy, modeled on the Week 7, 2026 winning
lineup (Burrow, Higgins and Chase with Flowers bringing it back; McCaffrey;
Pollard; Concepcion; Higbee; the Buccaneers). Each slot looks for the trait
that pick had before kickoff, across every Sunday 1pm and 4pm ET game, in
this order:

  QB:   the starter on a favored team in a game with an over/under of 46+,
        whose top two WRs each draw a 20%+ target share -- a passing game
        built on two receivers. The highest combined share of the two wins.
  WR1 and FLEX: those two WRs -- a double stack.
  WR2:  the opposing team's WR1 with a 25%+ share -- the bring-back.
  RB1:  a pass-catching lead back: 50%+ of his team's carries and a 20%+
        target share, on a team implied for 24+. The higher target share
        wins.
  RB2:  a cheap workhorse: $6,000 or less with 75%+ of his team's carries,
        in any game, however low its total. The higher carry share wins.
  WR3:  a WR with a 20%+ share at $6,000 or less from RB2's game -- the
        other team's WR1 first (the run-back), then RB2's teammate, then
        any game, by share.
  TE:   a punt TE: $5,000 or less, with a target inside the 10 this season,
        on a team implied for 26+. The higher target share wins.
  DEF:  $3,500 or less, facing a bottom-10 offense in sacks allowed per
        game, and no player the rules picked. Ranked by its own splash
        plays per game -- sacks plus two per interception, as FanDuel
        scores them.

Shares and carries are this season's (see vegas_strategy.get_player_season_stats);
a team's WR1 is its healthy WR with the highest target share. No more than
four players come from one game (the stack game has exactly four). Each
pick must leave enough under the cap to fill every slot still open at the
cheapest salaries on the slate, or the rule moves to its next candidate; a
rule with no candidate is skipped and its slot left to the optimizer. The
notes say which.

Picks are handed to the optimizer via the "required_group" column, as with
the other strategies.
"""
import pandas as pd

from alpha_rule import afternoon_games
from backtest import read_stats
from echo_rule import carry_shares
from matchup_filters import STATS_TEAM_WEEK_URL, build_eligibility
from optimizer import DEFAULT_SALARY_CAP
from real_stats import normalize_name
from vegas_strategy import _load_week_context, get_player_season_stats, implied_points

QB_MIN_TOTAL = 46.0
STACK_WR_MIN_SHARE = 0.20  # each of the QB's top two WRs
BRING_BACK_MIN_SHARE = 0.25
RB1_MIN_CARRY_SHARE = 0.50
RB1_MIN_TARGET_SHARE = 0.20
RB1_MIN_IMPLIED = 24.0
RB2_MAX_SALARY = 6000
RB2_MIN_CARRY_SHARE = 0.75
WR3_MAX_SALARY = 6000
WR3_MIN_SHARE = 0.20
TE_MAX_SALARY = 5000
TE_MIN_IMPLIED = 26.0
DEF_MAX_SALARY = 3500
MAX_FROM_GAME = 4
ROSTER = ["QB", "WR1", "FLEX", "WR2", "RB1", "RB2", "WR3", "TE", "DEF"]
SLOT_POSITION = {"QB": "QB", "WR1": "WR", "FLEX": "WR", "WR2": "WR", "RB1": "RB", "RB2": "RB",
                 "WR3": "WR", "TE": "TE", "DEF": "DEF"}


def defense_splash(team_week: pd.DataFrame) -> dict[str, float]:
    """Each defense's sacks plus two per interception, per game, from
    nflverse team weekly stats (regular season)."""
    games = team_week[team_week["season_type"] == "REG"]
    per_game = games.groupby("team")[["def_sacks", "def_interceptions"]].mean()
    return (per_game["def_sacks"] + 2 * per_game["def_interceptions"]).round(2).to_dict()


def apply_charlie_rule(
    pool: pd.DataFrame,
    odds: pd.DataFrame,
    player_stats: pd.DataFrame,
    bad_sack_teams: set[str],
    splash: dict[str, float],
    salary_cap: float = DEFAULT_SALARY_CAP,
) -> tuple[pd.DataFrame, list[str]]:
    """Restricts `pool` to the Sunday 1pm and 4pm games and tags each rule's
    pick in a "required_group" column. Returns (pool, notes).

    player_stats columns: match_key, team, position, carries, target_share,
    goal_line_targets. `bad_sack_teams` are the bottom-10 offenses in sacks
    allowed per game; `splash` is defense_splash's output.
    """
    games = afternoon_games(odds)
    games = games[games["home"].isin(pool["team"]) & games["away"].isin(pool["team"])]
    if games.empty:
        raise ValueError("No Sunday 1pm or 4pm games with Vegas lines match the uploaded slate.")
    implied, opponent, game_of, total = {}, {}, {}, {}
    for _, g in games.iterrows():
        implied.update(implied_points(g))
        opponent[g["home"]], opponent[g["away"]] = g["away"], g["home"]
        game_of[g["home"]] = game_of[g["away"]] = f"{g['away']} @ {g['home']}"
        total[g["home"]] = total[g["away"]] = g["total"]

    pool = pool[pool["team"].isin(opponent)].copy()
    pool["opponent"] = pool["team"].map(opponent)
    pool["required_group"] = None
    stats = player_stats.drop_duplicates(["match_key", "team", "position"])
    pool["match_key"] = pool["name"].apply(normalize_name)
    pool = pool.merge(stats[["match_key", "team", "position", "target_share", "goal_line_targets"]],
                      on=["match_key", "team", "position"], how="left")
    pool = pool.merge(carry_shares(stats), on=["match_key", "team"], how="left")
    for col in ("target_share", "goal_line_targets", "carry_share"):
        pool[col] = pool[col].fillna(0)

    notes: list[str] = []
    cheapest = {pos: pool.loc[pool["position"] == pos, "salary"].min() for pos in SLOT_POSITION.values()}

    def picked() -> pd.Series:
        return pool["required_group"].notna()

    def fits(i: int, label: str) -> bool:
        # Under the cap with every later slot at its cheapest, and under the
        # per-game limit.
        later = ROSTER[ROSTER.index(label) + 1:]
        open_cost = sum(cheapest[SLOT_POSITION[s]] for s in later if not (pool["required_group"] == s).any())
        spent = pool.loc[picked(), "salary"].sum()
        in_game = (pool.loc[picked(), "team"].map(game_of) == game_of[pool.loc[i, "team"]]).sum()
        return spent + pool.loc[i, "salary"] + open_cost <= salary_cap and in_game < MAX_FROM_GAME

    def take(ranked: list[int], label: str, why) -> int | None:
        # The first candidate that fits; notes say who was passed over.
        for n, i in enumerate(ranked):
            if pd.isna(pool.loc[i, "required_group"]) and fits(i, label):
                if n:
                    notes.append(f"{label}: {', '.join(pool.loc[ranked[:n], 'name'])} would break the "
                                 f"${salary_cap:,.0f} cap or the {MAX_FROM_GAME}-per-game limit; next best.")
                pool.loc[i, "required_group"] = label
                notes.append(f"{label} {pool.loc[i, 'name']} ({pool.loc[i, 'team']}, ${pool.loc[i, 'salary']:,}): "
                             f"{why(i)}.")
                return i
        notes.append(f"No {label} candidate fits; slot left to the optimizer." if ranked
                     else f"No {label} candidate; slot left to the optimizer.")
        return None

    def team_wrs(team: str) -> pd.DataFrame:
        return pool[(pool["team"] == team) & (pool["position"] == "WR")].sort_values(
            ["target_share", "projected_points"], ascending=False)

    # QB: a favored team in a 46+ game with two 20%+ WRs; the best duo first.
    duos = []
    for team in opponent:
        wrs = team_wrs(team)
        qbs = pool[(pool["team"] == team) & (pool["position"] == "QB")]
        if (len(wrs) >= 2 and not qbs.empty and total[team] >= QB_MIN_TOTAL
                and implied[team] > implied[opponent[team]]
                and (wrs["target_share"].iloc[:2] >= STACK_WR_MIN_SHARE).all()):
            duos.append((wrs["target_share"].iloc[:2].sum(), team))
    qb = None
    for _, team in sorted(duos, reverse=True):
        qbs = pool[(pool["team"] == team) & (pool["position"] == "QB")].sort_values(["salary", "projected_points"])
        wrs = team_wrs(team)
        qb = take([qbs.index[-1]], "QB", lambda i, t=team, w=wrs: (
            f"{t} favored in {game_of[t]} ({total[t]:g}), top two WRs {w['name'].iloc[0]} and "
            f"{w['name'].iloc[1]} at {w['target_share'].iloc[0]:.0%} and {w['target_share'].iloc[1]:.0%}"))
        if qb is not None:
            break
    if qb is None and not duos:
        notes.append(f"No favored team in a {QB_MIN_TOTAL:g}+ game has two WRs at a "
                     f"{STACK_WR_MIN_SHARE:.0%}+ target share; QB and the stack are left to the optimizer.")

    # WR1 and FLEX: the double stack. WR2: the bring-back.
    if qb is not None:
        qb_team = pool.loc[qb, "team"]
        stack = team_wrs(qb_team).index[:2]
        take([stack[0]], "WR1", lambda i: f"the QB's WR1, {pool.loc[i, 'target_share']:.0%} target share")
        take([stack[1]], "FLEX", lambda i: f"the QB's other top WR, {pool.loc[i, 'target_share']:.0%} target share")
        back = team_wrs(opponent[qb_team])
        if not back.empty and back["target_share"].iloc[0] >= BRING_BACK_MIN_SHARE:
            take([back.index[0]], "WR2", lambda i: (
                f"the bring-back, {opponent[qb_team]}'s WR1 at {pool.loc[i, 'target_share']:.0%}"))
        else:
            notes.append(f"No {opponent[qb_team]} WR1 at a {BRING_BACK_MIN_SHARE:.0%}+ target share; "
                         "WR2 left to the optimizer.")

    # RB1: a pass-catching lead back on a team implied for 24+.
    rb = pool[pool["position"] == "RB"]
    rb1 = rb[(rb["carry_share"] >= RB1_MIN_CARRY_SHARE) & (rb["target_share"] >= RB1_MIN_TARGET_SHARE)
             & (rb["team"].map(implied) >= RB1_MIN_IMPLIED)]
    take(list(rb1.sort_values(["target_share", "projected_points"], ascending=False).index), "RB1",
         lambda i: (f"{pool.loc[i, 'carry_share']:.0%} of the carries, {pool.loc[i, 'target_share']:.0%} "
                    f"target share, {pool.loc[i, 'team']} implied {implied[pool.loc[i, 'team']]:.1f}"))

    # RB2: a cheap workhorse, from any game.
    rb2 = rb[(rb["salary"] <= RB2_MAX_SALARY) & (rb["carry_share"] >= RB2_MIN_CARRY_SHARE)]
    rb2_pick = take(list(rb2.sort_values(["carry_share", "projected_points"], ascending=False).index), "RB2",
                    lambda i: (f"{pool.loc[i, 'carry_share']:.0%} of the carries in {game_of[pool.loc[i, 'team']]} "
                               f"({total[pool.loc[i, 'team']]:g})"))

    # WR3: a cheap 20%+ WR, the run-back from RB2's game first.
    cheap_wr = pool[(pool["position"] == "WR") & (pool["salary"] <= WR3_MAX_SALARY)
                    & (pool["target_share"] >= WR3_MIN_SHARE)]
    order, why = [], {}
    if rb2_pick is not None:
        rb2_team = pool.loc[rb2_pick, "team"]
        for team, kind in ((opponent[rb2_team], "the run-back"), (rb2_team, "RB2's teammate")):
            wrs = team_wrs(team)
            if not wrs.empty and wrs.index[0] in cheap_wr.index:
                order.append(wrs.index[0])
                why[wrs.index[0]] = f"{kind}, {team}'s WR1 in RB2's game"
    rest = cheap_wr.sort_values(["target_share", "projected_points"], ascending=False).index
    order += [i for i in rest if i not in order]
    take(order, "WR3", lambda i: f"{why.get(i, 'the best share elsewhere')}, {pool.loc[i, 'target_share']:.0%} target share")

    # TE: a punt with red-zone looks on a 26+ team.
    te = pool[(pool["position"] == "TE") & (pool["salary"] <= TE_MAX_SALARY) & (pool["goal_line_targets"] >= 1)
              & (pool["team"].map(implied) >= TE_MIN_IMPLIED)]
    take(list(te.sort_values(["target_share", "projected_points"], ascending=False).index), "TE",
         lambda i: (f"{pool.loc[i, 'target_share']:.0%} target share, {pool.loc[i, 'goal_line_targets']:.0f} "
                    f"targets inside the 10, {pool.loc[i, 'team']} implied {implied[pool.loc[i, 'team']]:.1f}"))

    # DEF: cheap, against a leaky line, facing none of the picks.
    offense_teams = set(pool.loc[picked() & (pool["position"] != "DEF"), "team"])
    defs = pool[(pool["position"] == "DEF") & (pool["salary"] <= DEF_MAX_SALARY)
                & pool["opponent"].isin(bad_sack_teams) & ~pool["opponent"].isin(offense_teams)]
    defs = defs.assign(splash=defs["team"].map(splash).fillna(0))
    take(list(defs.sort_values(["splash", "projected_points"], ascending=False).index), "DEF",
         lambda i: (f"faces {pool.loc[i, 'opponent']}, bottom-10 in sacks allowed; "
                    f"{splash.get(pool.loc[i, 'team'], 0):.2f} sacks + 2x INTs per game"))
    d = pool.index[pool["required_group"] == "DEF"]
    if len(d):
        faced = pool.loc[d[0], "opponent"]
        pool = pool[~((pool["team"] == faced) & (pool["position"] != "DEF") & pool["required_group"].isna())]

    pool["max_from_game"] = MAX_FROM_GAME
    pool = pool.drop(columns=["match_key", "target_share", "goal_line_targets", "carry_share"])
    return pool.reset_index(drop=True), notes


def build_charlie_pool(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fetches this week's lines and stats, then applies apply_charlie_rule."""
    week_info = build_eligibility()
    season, week = week_info["season"], week_info["week"]
    context = _load_week_context(season, week)
    stats = get_player_season_stats(season)
    splash = defense_splash(read_stats(STATS_TEAM_WEEK_URL.format(season)))
    pool, notes = apply_charlie_rule(pool, context["odds"], stats, set(context["bad_sack_teams"]), splash)
    notes.insert(0, f"Week {week} lines from ESPN (DraftKings).")
    return pool, notes
