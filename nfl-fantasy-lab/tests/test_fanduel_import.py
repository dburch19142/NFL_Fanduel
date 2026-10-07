"""Unit tests for parsing FanDuel player-pool CSV exports."""
import io

import pytest

from fanduel_import import load_fanduel_csv, FanDuelImportError

VALID_CSV = (
    "Id,Position,First Name,Nickname,Last Name,FPPG,Played,Salary,Game,Team,Opponent\n"
    "1,QB,Pat,Pat M,Mahomes,24.5,16,8200,KC@DEN,KC,DEN\n"
    "2,D,,,Chiefs,9.5,16,3400,KC@DEN,KC,DEN\n"
)


def test_parses_valid_export():
    df = load_fanduel_csv(io.StringIO(VALID_CSV))
    assert list(df["name"]) == ["Pat M", "Chiefs"]
    assert list(df["position"]) == ["QB", "DEF"]  # "D" normalized to "DEF"
    assert list(df["salary"]) == [8200, 3400]
    assert list(df["team"]) == ["KC", "KC"]


def test_normalizes_fanduel_team_abbreviations_to_nflverse():
    csv_text = (
        "Position,Nickname,Salary,Team\n"
        "QB,Trevor Lawrence,7900,JAC\n"
        "QB,Matthew Stafford,7800,LAR\n"
        "QB,Patrick Mahomes,8200,KC\n"
    )
    df = load_fanduel_csv(io.StringIO(csv_text))
    assert list(df["team"]) == ["JAX", "LA", "KC"]


def test_falls_back_to_first_last_name_when_no_nickname():
    csv_text = (
        "Position,First Name,Last Name,Salary,Team\n"
        "RB,Christian,McCaffrey,9500,SF\n"
    )
    df = load_fanduel_csv(io.StringIO(csv_text))
    assert df.loc[0, "name"] == "Christian McCaffrey"


def test_missing_required_column_raises_clear_error():
    csv_text = "Name,Cost,Squad\nSome Player,5000,ABC\n"
    with pytest.raises(FanDuelImportError) as exc_info:
        load_fanduel_csv(io.StringIO(csv_text))
    message = str(exc_info.value)
    assert "position" in message
    assert "salary" in message


def test_omits_any_player_with_an_injury_or_questionable_designation():
    csv_text = (
        "Position,Nickname,Salary,Team,Injury Indicator\n"
        "QB,Healthy Guy,8000,KC,\n"
        "RB,Questionable Guy,7000,SF,Q\n"
        "WR,Doubtful Guy,6000,DAL,D\n"
        "WR,Out Guy,5000,MIA,O\n"
        "TE,IR Guy,4000,BUF,IR\n"
    )
    df = load_fanduel_csv(io.StringIO(csv_text))
    assert list(df["name"]) == ["Healthy Guy"]
    assert df.attrs["excluded_injured_count"] == 4


def test_missing_injury_column_keeps_everyone():
    csv_text = "Position,Nickname,Salary,Team\nQB,Some Guy,8000,KC\n"
    df = load_fanduel_csv(io.StringIO(csv_text))
    assert len(df) == 1
    assert df.attrs["excluded_injured_count"] == 0
