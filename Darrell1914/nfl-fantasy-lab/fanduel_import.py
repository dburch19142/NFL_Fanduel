"""Parses a player-pool CSV exported from FanDuel's contest lineup builder
("Export to CSV" on the player list).

FanDuel's export format has drifted over time and can vary slightly by sport
and contest type, so column lookup here is tolerant: it matches headers
case-insensitively against a list of known aliases rather than assuming one
fixed schema. If your export doesn't match, the error message lists exactly
what headers were found so the aliases below can be extended.
"""
import io

import pandas as pd

POSITION_ALIASES = {
    "D": "DEF",
    "DST": "DEF",
    "D/ST": "DEF",
    "DEF": "DEF",
}

COLUMN_ALIASES = {
    "position": ["position", "roster position"],
    "salary": ["salary"],
    "team": ["team"],
    "nickname": ["nickname"],
    "first_name": ["first name"],
    "last_name": ["last name"],
    "fppg": ["fppg"],
}


class FanDuelImportError(ValueError):
    pass


def _find_column(columns_lower: dict, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in columns_lower:
            return columns_lower[alias]
    return None


def load_fanduel_csv(file) -> pd.DataFrame:
    """Loads a FanDuel player-pool export.

    `file` may be a path, an open file object, or raw bytes/str content.
    Returns a DataFrame with columns: name, position, team, salary, fppg.
    """
    if isinstance(file, (bytes, str)) and not _looks_like_path(file):
        df = pd.read_csv(io.StringIO(file if isinstance(file, str) else file.decode("utf-8")))
    else:
        df = pd.read_csv(file)

    columns_lower = {c.strip().lower(): c for c in df.columns}

    position_col = _find_column(columns_lower, COLUMN_ALIASES["position"])
    salary_col = _find_column(columns_lower, COLUMN_ALIASES["salary"])
    team_col = _find_column(columns_lower, COLUMN_ALIASES["team"])
    nickname_col = _find_column(columns_lower, COLUMN_ALIASES["nickname"])
    first_col = _find_column(columns_lower, COLUMN_ALIASES["first_name"])
    last_col = _find_column(columns_lower, COLUMN_ALIASES["last_name"])
    fppg_col = _find_column(columns_lower, COLUMN_ALIASES["fppg"])

    missing = []
    if position_col is None:
        missing.append("position")
    if salary_col is None:
        missing.append("salary")
    if team_col is None:
        missing.append("team")
    if nickname_col is None and not (first_col and last_col):
        missing.append("nickname (or first name + last name)")

    if missing:
        raise FanDuelImportError(
            "Couldn't find required column(s) "
            f"{missing} in the uploaded CSV. Columns found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    nickname = df[nickname_col].fillna("").str.strip() if nickname_col else pd.Series([""] * len(df))
    if first_col and last_col:
        full_name = (df[first_col].fillna("") + " " + df[last_col].fillna("")).str.strip()
    else:
        full_name = pd.Series([""] * len(df))
    # Defense rows often leave "Nickname" blank and put the team name in
    # "Last Name" instead, so fall back per-row rather than per-column.
    out["name"] = nickname.where(nickname != "", full_name)

    out["position"] = df[position_col].str.strip().str.upper().replace(POSITION_ALIASES)
    out["team"] = df[team_col].str.strip().str.upper()
    out["salary"] = pd.to_numeric(df[salary_col], errors="coerce")
    out["fppg"] = pd.to_numeric(df[fppg_col], errors="coerce") if fppg_col else 0.0

    out = out.dropna(subset=["name", "position", "team", "salary"])
    out = out[out["name"] != ""]
    return out.reset_index(drop=True)


def _looks_like_path(value: str) -> bool:
    return isinstance(value, str) and len(value) < 260 and ("/" in value or "\\" in value or value.endswith(".csv"))
