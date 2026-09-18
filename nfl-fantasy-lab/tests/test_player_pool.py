"""Unit tests for player_pool.py's matchup-eligibility filtering and upload handling."""
import io

import pandas as pd

import player_pool


def _sample_pool():
    return pd.DataFrame([
        {"name": "Eligible QB", "position": "QB", "team": "AAA", "salary": 8000, "projected_points": 20},
        {"name": "Ineligible QB", "position": "QB", "team": "BBB", "salary": 7500, "projected_points": 19},
        {"name": "Eligible WR", "position": "WR", "team": "AAA", "salary": 7000, "projected_points": 15},
        {"name": "Ineligible WR", "position": "WR", "team": "BBB", "salary": 6500, "projected_points": 14},
        {"name": "Some Defense", "position": "DEF", "team": "CCC", "salary": 3000, "projected_points": 8},
    ])


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
                "WR": {("eligible wr", "AAA")},
                "TE": set(),
            },
            "defense": {"pass": set(), "rush": set()},
        },
    )

    filtered, note = player_pool._apply_matchup_eligibility(_sample_pool())

    assert set(filtered["name"]) == {"Eligible QB", "Eligible WR", "Some Defense"}
    assert "Ineligible QB" not in set(filtered["name"])
    assert "week 2" in note


def test_matchup_eligibility_defense_always_passes_through(monkeypatch):
    monkeypatch.setattr(
        player_pool,
        "build_eligibility",
        lambda *a, **k: {
            "season": 2026, "week": 2,
            "eligible": {"QB": set(), "RB": set(), "WR": set(), "TE": set()},
            "defense": {"pass": set(), "rush": set()},
        },
    )
    filtered, _ = player_pool._apply_matchup_eligibility(_sample_pool())
    assert "Some Defense" in set(filtered["name"])


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
