"""Tests for the FanDuel assistant: the tools against the bundled sample pool,
and the tool-use loop against a scripted fake Claude client (no API calls)."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from app import app as flask_app
from fantasy_assistant import FantasyAssistant, FantasyTools, ToolError
from optimizer import PLAYERS_CSV


def sample_pool(strategy="matchup"):
    players = pd.read_csv(PLAYERS_CSV)
    players.attrs["strategy_notes"] = [f"{strategy} note"]
    return players, f"sample data ({strategy})"


@pytest.fixture
def tools():
    return FantasyTools(pool_builder=sample_pool)


def test_pool_filters_and_sorts_by_value(tools):
    result = tools.run("get_player_pool", {"strategy": "matchup", "position": "RB",
                                           "max_salary": 7000, "sort_by": "value", "limit": 5})
    players = result["players"]
    assert players and len(players) <= 5
    assert all(p["position"] == "RB" and p["salary"] <= 7000 for p in players)
    values = [p["value"] for p in players]
    assert values == sorted(values, reverse=True)
    assert result["strategy_notes"] == ["matchup note"]


def test_build_lineups_respects_cap_and_exclusions(tools):
    pool, _ = sample_pool()
    banned_team = pool["team"].value_counts().index[0]
    result = tools.run("build_lineups", {"strategy": "matchup", "count": 2, "exclude_teams": [banned_team]})
    assert 1 <= len(result["lineups"]) <= 2
    for lineup in result["lineups"]:
        assert lineup["total_salary"] <= 60000
        assert len(lineup["players"]) == 9
        assert banned_team not in {p["team"] for p in lineup["players"]}


def test_build_lineups_locks_player(tools):
    pool, _ = sample_pool()
    star = pool[pool["position"] == "WR"].sort_values("projected_points").iloc[0]["name"]
    result = tools.run("build_lineups", {"strategy": "matchup", "lock_player": star})
    assert star in {p["name"] for p in result["lineups"][0]["players"]}


def test_unknown_player_lock_is_a_tool_error(tools):
    with pytest.raises(ToolError, match="No player named"):
        tools.run("build_lineups", {"strategy": "matchup", "lock_player": "Nobody Atall"})


def test_pools_are_built_once_per_strategy():
    calls = []

    def counting_builder(strategy):
        calls.append(strategy)
        return sample_pool(strategy)

    tools = FantasyTools(pool_builder=counting_builder)
    tools.run("get_player_pool", {"strategy": "vegas"})
    tools.run("build_lineups", {"strategy": "vegas"})
    assert calls == ["vegas"]


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use(id_, name, args):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


class FakeClient:
    """Plays back scripted responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_ask_runs_tools_and_returns_final_answer(tools):
    client = FakeClient([
        SimpleNamespace(stop_reason="tool_use", content=[
            _tool_use("t1", "build_lineups", {"strategy": "matchup"}),
            _tool_use("t2", "get_standings", {}),
        ]),
        SimpleNamespace(stop_reason="end_turn", content=[_text("Here is your lineup.")]),
    ])
    assistant = FantasyAssistant(client=client, tools=tools)
    result = assistant.ask("Build me a lineup")

    assert result.text == "Here is your lineup."
    assert [c["name"] for c in result.tool_calls] == ["build_lineups", "get_standings"]
    # Both tool results go back together in one user message.
    tool_message = client.requests[1]["messages"][-1]
    assert tool_message["role"] == "user"
    assert [r["tool_use_id"] for r in tool_message["content"]] == ["t1", "t2"]
    lineup = json.loads(tool_message["content"][0]["content"])
    assert lineup["lineups"][0]["total_salary"] <= 60000
    assert client.requests[0]["fallbacks"] == "default"


def test_tool_failure_is_reported_to_claude_not_raised(tools):
    client = FakeClient([
        SimpleNamespace(stop_reason="tool_use", content=[
            _tool_use("t1", "build_lineups", {"strategy": "matchup", "lock_player": "Nobody Atall"}),
        ]),
        SimpleNamespace(stop_reason="end_turn", content=[_text("I couldn't find that player.")]),
    ])
    result = FantasyAssistant(client=client, tools=tools).ask("Lock Nobody Atall")
    error = client.requests[1]["messages"][-1]["content"][0]
    assert error["is_error"] is True and "No player named" in error["content"]
    assert result.tool_calls[0]["ok"] is False


def test_api_error_leaves_history_clean(tools):
    class Boom(FakeClient):
        def _create(self, **kwargs):
            raise RuntimeError("network down")

    assistant = FantasyAssistant(client=Boom([]), tools=tools)
    with pytest.raises(RuntimeError):
        assistant.ask("hello")
    assert assistant.messages == []


def test_ask_endpoint_rejects_empty_question():
    response = flask_app.test_client().post("/assistant/ask", json={"question": "  "})
    assert response.status_code == 400


def test_assistant_page_renders():
    response = flask_app.test_client().get("/assistant")
    assert response.status_code == 200
    assert b"Ask the Assistant" in response.data
