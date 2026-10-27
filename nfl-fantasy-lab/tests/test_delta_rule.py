"""Unit tests for the Delta Rule strategy (no network needed)."""
import pandas as pd

import player_pool
from delta_rule import apply_delta_rule
from optimizer import optimize_lineup

# All Sunday 1pm ET (Oct 4, 2026). AAA @ BBB (48) and CCC @ DDD (52) are the
# QB games; EEE @ FFF (38) is the low total; GGG @ HHH is a 43.
ODDS = pd.DataFrame([
    {"home": "BBB", "away": "AAA", "total": 48.0, "home_spread": -3.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "DDD", "away": "CCC", "total": 52.0, "home_spread": 4.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "FFF", "away": "EEE", "total": 38.0, "home_spread": -2.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "HHH", "away": "GGG", "total": 43.0, "home_spread": -1.0, "kickoff": "2026-10-04T17:00Z"},
])
TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
# Underdogs: AAA, DDD, EEE, GGG.


def _player(name, pos, team, salary, pts):
    return {"name": name, "position": pos, "team": team, "salary": salary, "projected_points": pts}


def _stat(name, pos, team, share=0.0, carries=0):
    return {"match_key": name.lower(), "position": pos, "team": team, "carries": carries, "target_share": share}


def _slate():
    players, stats = [], []
    for t in TEAMS:
        players += [_player(f"{t} QB", "QB", t, 7500, 18), _player(f"{t} QB2", "QB", t, 5000, 2),
                    _player(f"{t} RB1", "RB", t, 7000, 13), _player(f"{t} RB2", "RB", t, 4500, 6),
                    _player(f"{t} WR1", "WR", t, 7000, 13), _player(f"{t} WR2", "WR", t, 5500, 9),
                    _player(f"{t} TE", "TE", t, 4500, 7), _player(f"{t} Defense", "DEF", t, 3500, 7)]
        stats += [_stat(f"{t} RB1", "RB", t, carries=160), _stat(f"{t} RB2", "RB", t, carries=40),
                  _stat(f"{t} WR1", "WR", t, share=0.22), _stat(f"{t} WR2", "WR", t, share=0.15)]
    pool, stats = pd.DataFrame(players), pd.DataFrame(stats)
    # GGG splits its carries 60/40; two WR1s draw 25%+.
    stats.loc[stats["match_key"] == "ggg rb1", "carries"] = 60
    for key in ("fff wr1", "hhh wr1"):
        stats.loc[stats["match_key"] == key, "target_share"] = 0.27
    return pool, stats


def _run(pool=None, stats=None):
    base_pool, base_stats = _slate()
    return apply_delta_rule(base_pool if pool is None else pool, ODDS, base_stats if stats is None else stats)


def _groups(result, label):
    return set(result.loc[result["required_groups"].str.contains(label), "name"])


def test_qbs_are_starters_from_46_plus_games():
    result, notes = _run()
    assert set(result.loc[result["position"] == "QB", "name"]) == {"AAA QB", "BBB QB", "CCC QB", "DDD QB"}
    assert any(n.startswith("QB: a starter from a 46+ game") and "AAA @ BBB (48)" in n for n in notes)


def test_rbs_are_workhorses_only():
    result, _ = _run()
    rbs = set(result.loc[result["position"] == "RB", "name"])
    assert "GGG RB1" not in rbs and not any(n.endswith("RB2") for n in rbs)
    assert "AAA RB1" in rbs  # 80% of the carries


def test_fewer_than_two_workhorses_keeps_every_rb():
    _, stats = _slate()
    stats.loc[stats["position"] == "RB", "carries"] = 100
    result, notes = _run(stats=stats)
    assert (result["position"] == "RB").sum() == 16
    assert any("every RB stays" in n for n in notes)


def test_defenses_are_underdogs_only():
    result, _ = _run()
    defs = set(result.loc[result["position"] == "DEF", "team"])
    assert defs == {"AAA", "DDD", "EEE", "GGG"}


def test_requirement_groups():
    result, _ = _run()
    assert _groups(result, "WR1_25") == {"FFF WR1", "HHH WR1"}
    low = _groups(result, "LOW_TOTAL")
    assert {"EEE WR1", "FFF TE", "EEE RB1"} <= low and not any(n.startswith(("AAA", "GGG")) for n in low)


def test_lineup_meets_every_delta_requirement():
    result, _ = _run()
    lineup = optimize_lineup(60000, result)
    qb = lineup[lineup["position"] == "QB"].iloc[0]
    skill = lineup[lineup["position"].isin(["RB", "WR", "TE"])]
    assert (skill["team"] == qb["opponent"]).any()  # bring-back
    assert skill["name"].isin(_groups(result, "WR1_25")).any()
    assert skill["name"].isin(_groups(result, "LOW_TOTAL")).any()
    assert lineup.loc[lineup["position"] == "DEF", "team"].iloc[0] in {"AAA", "DDD", "EEE", "GGG"}
    games = ["|".join(sorted(pair)) for pair in zip(lineup["team"], lineup["opponent"])]
    assert pd.Series(games).value_counts().max() <= 4


def test_bring_back_constraint_changes_the_lineup():
    # CCC's QB projects best, but every DDD skill player projects 1 point.
    pool, stats = _slate()
    pool.loc[pool["name"] == "CCC QB", "projected_points"] = 40
    pool.loc[(pool["team"] == "DDD") & pool["position"].isin(["RB", "WR", "TE"]), "projected_points"] = 1
    result, _ = _run(pool=pool, stats=stats)
    free = optimize_lineup(60000, result.assign(bring_back=False))
    held = optimize_lineup(60000, result)
    for lineup in (free, held):
        assert lineup.loc[lineup["position"] == "QB", "name"].iloc[0] == "CCC QB"
    assert not (free["team"] == "DDD").any()
    assert ((held["team"] == "DDD") & held["position"].isin(["RB", "WR", "TE"])).any()


def test_delta_strategy_falls_back_to_matchup_filters(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("odds unavailable")

    monkeypatch.setattr(player_pool, "build_delta_pool", _boom)
    monkeypatch.setattr(player_pool, "build_eligibility", _boom)
    pool, _ = _slate()
    result, note = player_pool._apply_delta_strategy(pool)
    assert "Delta Rule unavailable (odds unavailable)" in note
    assert len(result) == len(pool)
