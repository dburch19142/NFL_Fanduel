"""FanDuel assistant: Claude answers questions about this week's slate by
calling the same player-pool, optimizer and standings code the web pages use,
plus the NFL Assistant's injury, depth chart, schedule and stats tools when
nfl-assistant is installed alongside (see nfl_data.py).

Claude never sees a projection or salary it didn't get from a tool, so its
lineups are the optimizer's lineups and its numbers are the pool's numbers.
Needs ANTHROPIC_API_KEY (or an `ant auth login` profile) to reach Claude.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import anthropic
import pandas as pd

import nfl_data

from optimizer import (
    DEFAULT_SALARY_CAP,
    InfeasibleLineupError,
    PlayerNotFoundError,
    optimize_lineups,
)
from player_pool import (
    STRATEGY_ALPHA,
    STRATEGY_BETA,
    STRATEGY_ECHO,
    STRATEGY_CHARLIE,
    STRATEGY_DELTA,
    STRATEGY_COMBINED,
    STRATEGY_MATCHUP,
    STRATEGY_VEGAS,
    UPLOAD_PATH,
    build_player_pool,
    has_uploaded_pool,
)
from real_stats import normalize_name

MODEL = os.environ.get("ANTHROPIC_MODEL") or "claude-opus-5-5"
EFFORT = os.environ.get("ANTHROPIC_EFFORT") or "medium"
MAX_TOOL_ROUNDS = 10
MAX_LINEUPS = 10
TEAMS_CSV = "data/teams.csv"

STRATEGIES = [STRATEGY_MATCHUP, STRATEGY_VEGAS, STRATEGY_COMBINED, STRATEGY_ALPHA, STRATEGY_BETA, STRATEGY_ECHO, STRATEGY_CHARLIE, STRATEGY_DELTA]
# The NFL Assistant tools offered to Claude, when nfl-assistant is available.
NFL_TOOLS = ["get_injuries", "get_depth_chart", "get_player", "get_games", "get_top_totals", "get_team_leaders"]
MAX_LISTED_EXCLUSIONS = 40

POOL_COLUMNS = ["name", "position", "team", "salary", "projected_points", "opponent", "required_group"]

SYSTEM_PROMPT = """\
You are the FanDuel assistant inside NFL Fantasy Lab, a tool for building FanDuel NFL \
classic lineups (1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX from RB/WR/TE, 1 DEF, $60,000 cap).

Where your facts come from
Salaries, projections, injuries and matchups change every week and your own NFL knowledge \
is out of date. For anything about this slate, call the tools and use only what they \
return; never fill a gap from memory. If a player isn't in the pool, say so and use \
find_player to explain why (not on the slate, or filtered out by the strategy). General \
football or DFS knowledge (what a stack is, why you'd fade a defense) needs no tool.

Injuries, depth charts, schedule and stats
Every pool already has this week's injury report applied: anyone Out, Doubtful, \
Questionable or on injured reserve is removed, and tool results list them under \
removed_by_injury_report. When a player is missing, check there first. If the \
get_injuries, get_depth_chart, get_player, get_games, get_top_totals and \
get_team_leaders tools are available, use them to explain a pick (who replaces an \
injured starter, recent production, the Vegas total) and say how old the data is from \
data_freshness when it is stale. If pool_source says the live injury report is \
unavailable, warn that only the FanDuel export's injury column was applied.

Lineups
Always build lineups with build_lineups. Never assemble or edit a lineup by hand, because \
only the optimizer enforces the cap, roster slots, QB stack and the DEF rule. To honor a \
request like "no Chiefs" or "lock Bijan", pass exclusions or a lock to the tool. Show each \
lineup as a compact list (slot, player, team, salary, projection) with its total salary and \
projected points.

Strategies
matchup: full slate filtered by defensive matchups (the default). vegas: players from the \
top three Vegas games by total. combined: the Vegas picks plus matchup filters for the \
other slots. alpha: the Alpha Rule, the three highest over/under Sunday 1pm and 4pm ET games only. \
beta: RBs only from teams with a top-10 rushing offense facing a bottom-10 rushing defense \
on Sunday at 1pm or 4pm ET, matchup filters for every other slot. echo: the Echo Rule, the \
Alpha Rule with a QB from a top-3 over/under game, Beta's RBs, WR1/WR2 at 25%+ target share, a 15-18% WR3, 46+ totals only, no committee backs, and a FLEX that needs a real role, a good \
game script and 2.5+ points per $1,000 of salary. charlie: the Charlie Rule, a QB with two 20%+ WRs \
in a 46+ game (double stack plus a 25%+ bring-back), a pass-catching lead back, a cheap workhorse \
with a cheap 20%+ WR from his game, a punt TE and a cheap DEF facing a leaky line. delta: the Delta Rule, a QB from a 46+ game with \
a bring-back from his opponent, workhorse RBs (70%+ of carries), a 25%+ team WR1, a player from \
a sub-40 game and an underdog DEF. If the user names no \
strategy, use matchup. Tool results include strategy_notes and pool_source; mention them \
when they explain a pick or a missing player.

Style
Short and scannable. Lead with the answer. Value means projected points per $1,000 of \
salary. These are projections, not guarantees: don't promise outcomes or tell anyone how \
much to wager.

Today's date is {today}.
"""

TOOLS = [
    {
        "name": "get_player_pool",
        "description": (
            "List players in this week's pool for a strategy, with salary, projected points, "
            "and value (points per $1,000). Filter by position, team or salary, and sort. "
            "Use it to answer 'best value RBs', 'cheap TEs', 'who's in the Vegas pool'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": STRATEGIES},
                "position": {"type": "string", "enum": ["QB", "RB", "WR", "TE", "DEF"]},
                "team": {"type": "string", "description": "Team abbreviation, e.g. KC."},
                "min_salary": {"type": "integer"},
                "max_salary": {"type": "integer"},
                "sort_by": {"type": "string", "enum": ["projected_points", "value", "salary"]},
                "limit": {"type": "integer", "description": "Max rows, default 15, max 60."},
            },
            "required": ["strategy"],
            "additionalProperties": False,
        },
    },
    {
        "name": "build_lineups",
        "description": (
            "Run the lineup optimizer and return up to `count` distinct lineups, best first. "
            "Supports locking one player and excluding players or whole teams."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": STRATEGIES},
                "count": {"type": "integer", "description": f"1-{MAX_LINEUPS}, default 1."},
                "salary_cap": {"type": "integer", "description": f"Default {DEFAULT_SALARY_CAP}."},
                "lock_player": {"type": "string", "description": "Name of a player every lineup must include."},
                "lock_team": {"type": "string", "description": "Team of the locked player, if the name is ambiguous."},
                "exclude_players": {"type": "array", "items": {"type": "string"}},
                "exclude_teams": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["strategy"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_player",
        "description": (
            "Look a player up by name: whether they're on the uploaded FanDuel slate, their "
            "salary and FPPG there, and which strategy pools include them with what projection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_standings",
        "description": "Current NFL standings: wins, losses, ties, points for and against.",
        "input_schema": {
            "type": "object",
            "properties": {"team": {"type": "string", "description": "Optional full team name filter."}},
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    pass


def _records(df: pd.DataFrame) -> list[dict]:
    cols = [c for c in POOL_COLUMNS if c in df.columns]
    out = df[cols].copy()
    out["value"] = (out["projected_points"] / (out["salary"] / 1000)).round(2)
    return json.loads(out.to_json(orient="records"))


class FantasyTools:
    """Runs tool calls. Pools are cached per strategy because building one can
    hit nflverse and odds feeds; create a fresh instance to pick up new data."""

    def __init__(self, pool_builder: Callable[..., tuple[pd.DataFrame, str]] = build_player_pool):
        self._pool_builder = pool_builder
        self._pools: dict[str, tuple[pd.DataFrame, str]] = {}

    def pool(self, strategy: str) -> tuple[pd.DataFrame, str]:
        if strategy not in STRATEGIES:
            raise ToolError(f"Unknown strategy {strategy!r}. Use one of {STRATEGIES}.")
        if strategy not in self._pools:
            self._pools[strategy] = self._pool_builder(strategy=strategy)
        return self._pools[strategy]

    def _context(self, strategy: str) -> dict:
        players, source = self.pool(strategy)
        removed = list(players.attrs.get("injury_exclusions", []))
        context = {
            "strategy": strategy,
            "pool_source": source,
            "strategy_notes": list(players.attrs.get("strategy_notes", [])),
            "pool_size": len(players),
            "removed_by_injury_report": removed[:MAX_LISTED_EXCLUSIONS],
        }
        if len(removed) > MAX_LISTED_EXCLUSIONS:
            context["removed_by_injury_report_total"] = len(removed)
        return context

    def run(self, name: str, args: dict) -> dict:
        if name in NFL_TOOLS:
            return nfl_data.run_tool(name, args)
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolError(f"Unknown tool {name!r}.")
        return handler(**args)

    def _tool_get_player_pool(self, strategy, position=None, team=None, min_salary=None,
                              max_salary=None, sort_by="projected_points", limit=15):
        players, _ = self.pool(strategy)
        df = players
        if position:
            df = df[df["position"] == position.upper()]
        if team:
            df = df[df["team"].str.upper() == team.upper()]
        if min_salary is not None:
            df = df[df["salary"] >= min_salary]
        if max_salary is not None:
            df = df[df["salary"] <= max_salary]
        rows = _records(df)
        key = sort_by if sort_by in ("projected_points", "value", "salary") else "projected_points"
        rows.sort(key=lambda r: r[key], reverse=True)
        limit = max(1, min(int(limit or 15), 60))
        return {**self._context(strategy), "matches": len(rows), "players": rows[:limit]}

    def _tool_build_lineups(self, strategy, count=1, salary_cap=DEFAULT_SALARY_CAP, lock_player=None,
                            lock_team=None, exclude_players=None, exclude_teams=None):
        players, _ = self.pool(strategy)
        df = players
        if exclude_teams:
            teams = {t.upper() for t in exclude_teams}
            df = df[~df["team"].str.upper().isin(teams)]
        if exclude_players:
            names = {normalize_name(n) for n in exclude_players}
            df = df[~df["name"].apply(normalize_name).isin(names)]
        count = max(1, min(int(count or 1), MAX_LINEUPS))
        try:
            results = optimize_lineups(
                float(salary_cap or DEFAULT_SALARY_CAP), df.reset_index(drop=True), count=count,
                required_name=lock_player or None, required_team=lock_team or None,
            )
        except (InfeasibleLineupError, PlayerNotFoundError) as exc:
            raise ToolError(str(exc)) from exc
        lineups = [
            {
                "players": _records(lineup),
                "total_salary": int(lineup["salary"].sum()),
                "total_projected_points": round(float(lineup["projected_points"].sum()), 1),
            }
            for lineup in results
        ]
        return {**self._context(strategy), "lineups": lineups}

    def _tool_find_player(self, name):
        target = normalize_name(name)
        result: dict = {"query": name}
        if has_uploaded_pool():
            slate = pd.read_csv(UPLOAD_PATH)
            hits = slate[slate["name"].apply(normalize_name) == target]
            result["on_uploaded_slate"] = json.loads(hits.to_json(orient="records"))
        else:
            result["on_uploaded_slate"] = None
            result["note"] = "No FanDuel export uploaded; pools use the bundled sample data."
        in_pools = {}
        for strategy in STRATEGIES:
            try:
                players, _ = self.pool(strategy)
            except Exception as exc:  # one broken feed shouldn't hide the others
                in_pools[strategy] = f"pool unavailable: {exc}"
                continue
            hits = players[players["name"].apply(normalize_name) == target]
            in_pools[strategy] = _records(hits) or "not in this pool"
            for removed in players.attrs.get("injury_exclusions", []):
                if normalize_name(removed["name"]) == target:
                    result["removed_by_injury_report"] = removed
        result["strategy_pools"] = in_pools
        return result

    def _tool_get_standings(self, team=None):
        teams = pd.read_csv(TEAMS_CSV)
        if team:
            teams = teams[teams["team"].str.lower().str.contains(team.lower())]
        return {"teams": json.loads(teams.to_json(orient="records"))}


@dataclass
class TurnResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str | None = None


class FantasyAssistant:
    """One conversation. Keep one per browser session."""

    def __init__(self, client: anthropic.Anthropic | None = None, tools: FantasyTools | None = None):
        self.client = client or anthropic.Anthropic()
        self.tools = tools or FantasyTools()
        self.tool_specs = TOOLS + nfl_data.nfl_tool_schemas(NFL_TOOLS)
        self.messages: list[dict] = []

    def _request(self):
        return self.client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT.format(today=date.today().isoformat()),
            tools=self.tool_specs,
            messages=self.messages,
            output_config={"effort": EFFORT},
            # On a safety decline, rerun on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def _run_tools(self, blocks, trace: list[dict]) -> list[dict]:
        results = []
        for block in blocks:
            call = {"name": block.name, "input": block.input, "ok": True}
            try:
                content = json.dumps(self.tools.run(block.name, dict(block.input)), default=str)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
            except Exception as exc:
                call["ok"] = False
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Error: {exc}", "is_error": True})
            trace.append(call)
        return results

    def ask(self, user_text: str) -> TurnResult:
        """Send one user message and loop until Claude finishes its answer."""
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        trace: list[dict] = []
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = self._request()
                if response.stop_reason == "refusal":
                    del self.messages[checkpoint:]
                    return TurnResult("I can't help with that request.", trace, "refusal")
                # Keep the full content (thinking and tool_use blocks) for the next request.
                self.messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason == "tool_use":
                    calls = [b for b in response.content if b.type == "tool_use"]
                    # All results go back in a single user message.
                    self.messages.append({"role": "user", "content": self._run_tools(calls, trace)})
                    continue
                text = "\n".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n(Answer cut off at the length limit. Ask me to continue.)"
                return TurnResult(text, trace, response.stop_reason)
            del self.messages[checkpoint:]
            return TurnResult("That took too many lookups. Try a narrower question.", trace, "tool_limit")
        except Exception:
            # Drop the half-finished turn so the history stays valid for the next question.
            del self.messages[checkpoint:]
            raise


def friendly_error(exc: Exception) -> str:
    """Turn an SDK exception into one line a user can act on."""
    if "Could not resolve authentication method" in str(exc):
        return "No Claude API key found. Set ANTHROPIC_API_KEY before starting the app."
    if isinstance(exc, anthropic.AuthenticationError):
        return "Claude rejected the API key. Check ANTHROPIC_API_KEY."
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by the Claude API. Wait a moment and try again."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Claude API error {exc.status_code}: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Claude API. Check your network connection."
    return f"Unexpected error: {exc}"
