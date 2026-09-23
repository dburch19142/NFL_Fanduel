"""Unit tests for player_pool.py's matchup-eligibility filtering and upload handling."""
import io

import pandas as pd

import player_pool


def _sample_pool():
    # Three eligible WRs and one eligible TE -- enough to clear
    # ROSTER_SLOTS["WR"] (3) and ROSTER_SLOTS["TE"] (1) on their own, so the
    # too-thin relaxation (tested separately below) doesn't kick in and mask
    # what these base filtering tests check. _ELIGIBLE_WR_TE below names the
    # (name, team) keys the tests mark eligible to keep both in sync.
    return pd.DataFrame([
        {"name": "Eligible QB", "position": "QB", "team": "AAA", "salary": 8000, "projected_points": 20},
        {"name": "Ineligible QB", "position": "QB", "team": "BBB", "salary": 7500, "projected_points": 19},
        {"name": "Eligible WR", "position": "WR", "team": "AAA", "salary": 7000, "projected_points": 15},
        {"name": "Eligible WR Two", "position": "WR", "team": "AAA", "salary": 6800, "projected_points": 14},
        {"name": "Eligible WR Three", "position": "WR", "team": "AAA", "salary": 6600, "projected_points": 13},
        {"name": "Ineligible WR", "position": "WR", "team": "BBB", "salary": 6500, "projected_points": 14},
        {"name": "Eligible TE", "position": "TE", "team": "AAA", "salary": 5000, "projected_points": 10},
        {"name": "Ineligible TE", "position": "TE", "team": "BBB", "salary": 4500, "projected_points": 9},
        {"name": "Some Defense", "position": "DEF", "team": "CCC", "salary": 3000, "projected_points": 8},
    ])


_ELIGIBLE_WR = {("eligible wr", "AAA"), ("eligible wr two", "AAA"), ("eligible wr three", "AAA")}
_ELIGIBLE_TE = {("eligible te", "AAA")}


def test_matchup_eligibility_filters_out_ineligible_skill_players(monkeypatch):
    monkeypatch.setattr(
        player_pool,
        "build_eligibility",
        lambda *a, **k: {
            "season": 2026,
            "week": 2,
            "eligible": {
                "QB": {("eligible qb", "AAA")},
                "RB": set(),
                "WR": _ELIGIBLE_WR,
                "TE": _ELIGIBLE_TE,
            },
        },
    )

    filtered, note = player_pool._apply_matchup_eligibility(_sample_pool())

    assert set(filtered["name"]) == {
        "Eligible QB", "Eligible WR", "Eligible WR Two", "Eligible WR Three", "Eligible TE", "Some Defense",
    }
    assert "Ineligible QB" not in set(filtered["name"])
    assert "Ineligible WR" not in set(filtered["name"])
    assert "Ineligible TE" not in set(filtered["name"])
    assert "week 2" in note
    assert "relaxed" not in note  # WR and TE both had enough eligible players, no fallback needed


def test_matchup_eligibility_defense_always_passes_through(monkeypatch):
    monkeypatch.setattr(
        player_pool,
        "build_eligibility",
        lambda *a, **k: {
            "season": 2026, "week": 2,
            "eligible": {"QB": set(), "RB": set(), "WR": set(), "TE": set()},
        },
    )
    filtered, _ = player_pool._apply_matchup_eligibility(_sample_pool())
    assert "Some Defense" in set(filtered["name"])


def test_matchup_eligibility_relaxes_wr_when_too_few_eligible(monkeypatch):
    monkeypatch.setattr(
        player_pool,
        "build_eligibility",
        lambda *a, **k: {
            "season": 2026, "week": 2,
            "eligible": {
                "QB": {("eligible qb", "AAA")},
                "RB": set(),
                "WR": {("eligible wr", "AAA")},  # only 1 eligible WR, roster needs 3
                "TE": _ELIGIBLE_TE,
            },
        },
    )

    filtered, note = player_pool._apply_matchup_eligibility(_sample_pool())

    # WR rule relaxed entirely -- every WR passes through, eligible or not.
    assert {"Eligible WR", "Eligible WR Two", "Eligible WR Three", "Ineligible WR"} <= set(filtered["name"])
    # TE had enough eligible players, so it keeps the strict rule.
    assert "Ineligible TE" not in set(filtered["name"])
    # QB keeps the strict rule regardless -- it isn't a relaxable position.
    assert "Ineligible QB" not in set(filtered["name"])
    assert "WR rule relaxed this week" in note
    assert "TE rule relaxed" not in note


def test_matchup_eligibility_does_not_relax_qb(monkeypatch):
    monkeypatch.setattr(
        player_pool,
        "build_eligibility",
        lambda *a, **k: {
            "season": 2026, "week": 2,
            "eligible": {
                "QB": set(),  # zero eligible QBs, well below the roster's 1 needed
                "RB": set(),
                "WR": _ELIGIBLE_WR,
                "TE": _ELIGIBLE_TE,
            },
        },
    )

    filtered, note = player_pool._apply_matchup_eligibility(_sample_pool())

    # QB isn't in RELAXABLE_POSITIONS, so it stays strict even when thin.
    assert "Eligible QB" not in set(filtered["name"])
    assert "relaxed" not in note


def test_matchup_eligibility_falls_back_gracefully_when_unavailable(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(player_pool, "build_eligibility", _boom)

    pool = _sample_pool()
    filtered, note = player_pool._apply_matchup_eligibility(pool)

    assert len(filtered) == len(pool)  # unfiltered fallback, nobody dropped
    assert "unavailable" in note


class _FakeUpload:
    def __init__(self, csv_text: str):
        self.stream = io.BytesIO(csv_text.encode("utf-8"))


def test_save_uploaded_csv_refreshes_sample_data(tmp_path, monkeypatch):
    monkeypatch.setattr(player_pool, "UPLOAD_PATH", str(tmp_path / "fanduel_latest.csv"))
    monkeypatch.setattr(player_pool, "PLAYERS_CSV", str(tmp_path / "players.csv"))

    csv_text = (
        "Position,Salary,Team,Nickname,FPPG,Injury Indicator\n"
        "QB,8000,AAA,Healthy Guy,20.5,\n"
        "WR,4000,BBB,Hurt Guy,15.0,Q\n"
        "TE,4500,CCC,No Fppg Guy,,\n"
    )

    player_pool.save_uploaded_csv(_FakeUpload(csv_text))

    sample = pd.read_csv(player_pool.PLAYERS_CSV)
    assert set(sample["name"]) == {"Healthy Guy", "No Fppg Guy"}  # injured player excluded
    assert list(sample.columns) == ["name", "position", "team", "salary", "projected_points"]

    no_fppg_row = sample[sample["name"] == "No Fppg Guy"].iloc[0]
    assert no_fppg_row["projected_points"] == 0  # blank FPPG becomes 0, not NaN
