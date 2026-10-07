"""Unit tests for the Top-3 Vegas games strategy (no network needed)."""
import pandas as pd

import player_pool
from optimizer import optimize_lineup
from vegas_strategy import apply_vegas_rules, implied_points, parse_scoreboard_odds, top_games

# Four games; the lowest-total one (GGG @ HHH) should be dropped.
ODDS = pd.DataFrame([
    {"home": "BBB", "away": "AAA", "total": 52.0, "home_spread": -3.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "DDD", "away": "CCC", "total": 49.0, "home_spread": 9.0, "kickoff": "2026-10-04T17:00Z"},
    {"home": "FFF", "away": "EEE", "total": 48.0, "home_spread": -1.0, "kickoff": "2026-10-04T20:25Z"},
    {"home": "HHH", "away": "GGG", "total": 40.0, "home_spread": -3.0, "kickoff": "2026-10-04T17:00Z"},
])


def _player(name, pos, team, salary=5000, pts=10.0):
    return {"name": name, "position": pos, "team": team, "salary": salary, "projected_points": pts}


def _stat(name, pos, team, pass_attempts=0, rush=0.0, share=0.0, goal_line=0):
    return {"match_key": name.lower(), "position": pos, "team": team, "pass_attempts": pass_attempts,
            "rushing_yards_pg": rush, "target_share": share, "goal_line_targets": goal_line}


def _pool_and_stats():
    players, stats = [], []
    for team in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"):
        players += [
            _player(f"{team} QB", "QB", team, 7000, 18),
            _player(f"{team} RB1", "RB", team, 6000, 12),
            _player(f"{team} RB2", "RB", team, 5000, 8),
            _player(f"{team} WR1", "WR", team, 7000, 14),
            _player(f"{team} WR2", "WR", team, 6000, 11),
            _player(f"{team} WR3", "WR", team, 4500, 6),
            _player(f"{team} TE", "TE", team, 4500, 7),
            _player(f"{team} Defense", "DEF", team, 3500, 7),
        ]
        stats += [
            _stat(f"{team} QB", "QB", team, pass_attempts=100, rush=10),
            _stat(f"{team} WR1", "WR", team, share=0.27),
            _stat(f"{team} WR2", "WR", team, share=0.20),
            _stat(f"{team} WR3", "WR", team, share=0.10),
            _stat(f"{team} TE", "TE", team, goal_line=1 if team != "CCC" else 0),
        ]
    return pd.DataFrame(players), pd.DataFrame(stats)


def _run(pool=None, stats=None, bad_pass=("BBB", "AAA"), bad_rush=(), bad_sacks=()):
    base_pool, base_stats = _pool_and_stats()
    return apply_vegas_rules(
        base_pool if pool is None else pool,
        ODDS,
        base_stats if stats is None else stats,
        pass_yds_allowed={"AAA": 250, "BBB": 260, "CCC": 270, "DDD": 200, "EEE": 210, "FFF": 220},
        bad_pass_teams=set(bad_pass),
        bad_rush_teams=set(bad_rush),
        bad_sack_teams=set(bad_sacks),
    )


def _group(pool, label):
    return set(pool.loc[pool["required_group"] == label, "name"])


def test_parse_scoreboard_odds_maps_espn_abbreviations():
    scoreboard = {"events": [{
        "date": "2026-10-04T17:00Z",
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"abbreviation": "WSH"}},
                {"homeAway": "away", "team": {"abbreviation": "LAR"}},
            ],
            "odds": [{"overUnder": 47.5, "spread": -3.5}],
        }],
    }]}
    odds = parse_scoreboard_odds(scoreboard)
    assert odds.iloc[0].to_dict() == {
        "home": "WAS", "away": "LA", "total": 47.5, "home_spread": -3.5, "kickoff": "2026-10-04T17:00Z",
    }


def test_implied_points_favors_the_favorite():
    game = pd.Series({"home": "BUF", "away": "NE", "total": 48.5, "home_spread": -7.0})
    assert implied_points(game) == {"BUF": 27.75, "NE": 20.75}


def test_top_games_only_counts_games_on_the_slate():
    slate = {"AAA", "BBB", "CCC", "DDD", "GGG", "HHH"}  # EEE/FFF not on this slate
    games = top_games(ODDS, slate)
    assert list(games["home"]) == ["BBB", "DDD", "HHH"]


def test_pool_restricted_to_top_three_games():
    pool, notes = _run()
    assert set(pool["team"]) == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}
    assert "AAA @ BBB (52)" in notes[0]


def test_qb_is_best_runner_facing_a_bad_pass_defense():
    pool, stats = _pool_and_stats()
    stats.loc[stats["match_key"] == "bbb qb", "rushing_yards_pg"] = 40
    result, _ = _run(pool, stats)
    assert _group(result, "QB") == {"BBB QB"}


def test_qb_falls_back_to_worst_pass_defense_faced():
    result, notes = _run(bad_pass=())
    # CCC allows the most pass yards, so DDD's QB (facing CCC) is picked.
    assert _group(result, "QB") == {"DDD QB"}
    assert any("worst" in n for n in notes)


def test_stack_bring_back_and_other_game_wr2s():
    pool, stats = _pool_and_stats()
    stats.loc[stats["match_key"] == "bbb qb", "rushing_yards_pg"] = 40
    result, _ = _run(pool, stats)
    assert _group(result, "WR1") == {"BBB WR1"}
    assert _group(result, "WR2") == {"AAA WR1"}
    assert _group(result, "WR3") == {"CCC WR2", "DDD WR2", "EEE WR2", "FFF WR2"}


def test_wr1_falls_back_when_no_teammate_has_25_percent_share():
    pool, stats = _pool_and_stats()
    stats.loc[stats["match_key"] == "bbb qb", "rushing_yards_pg"] = 40
    stats.loc[stats["match_key"] == "bbb wr1", "target_share"] = 0.20
    result, notes = _run(pool, stats)
    assert _group(result, "WR1") == {"BBB WR1"}
    assert any("25%+ target share" in n for n in notes)


def test_backup_qb_detected_when_season_leader_missing():
    pool, stats = _pool_and_stats()
    pool = pool[pool["name"] != "DDD QB"]
    pool = pd.concat([pool, pd.DataFrame([_player("DDD Backup", "QB", "DDD", 6000, 3)])])
    result, notes = _run(pool, stats)
    assert any("Starting a backup QB: DDD (DDD Backup)" in n for n in notes)
    # DDD's RBs qualify for the RB rule because of the backup QB.
    assert _group(result, "RB") == {"DDD RB1", "DDD RB2"}
    # CCC's defense faces that backup.
    assert _group(result, "DEF") == {"CCC Defense"}


def test_rb_rule_includes_bad_rush_defense_matchups():
    result, _ = _run(bad_rush=("FFF",))
    assert _group(result, "RB") == {"EEE RB1", "EEE RB2"}


def test_te_limited_to_goal_line_targets():
    result, _ = _run()
    assert "CCC TE" not in set(result["name"])
    assert "DDD TE" in set(result["name"])


def test_def_prefers_low_implied_opponent_and_avoids_qb_game():
    result, notes = _run()
    # DDD is a 9-point home underdog: implied 20.0, under ~20 -> CCC's defense.
    assert _group(result, "DEF") == {"CCC Defense"}
    assert not {"AAA Defense", "BBB Defense"} & _group(result, "DEF")
    assert any("implied 20.0" in n for n in notes)


def test_optimizer_honors_required_groups():
    result, _ = _run()
    lineup = optimize_lineup(60000, result)
    picked = set(lineup["name"])
    for label in result["required_group"].dropna().unique():
        assert picked & _group(result, label), label


def test_vegas_strategy_falls_back_to_matchup_filters(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("odds unavailable")

    monkeypatch.setattr(player_pool, "build_vegas_pool", _boom)
    monkeypatch.setattr(player_pool, "build_eligibility", _boom)
    pool, _ = _pool_and_stats()
    result, note = player_pool._apply_vegas_strategy(pool)
    assert "Top-3 Vegas games strategy unavailable (odds unavailable)" in note
    assert len(result) == len(pool)
