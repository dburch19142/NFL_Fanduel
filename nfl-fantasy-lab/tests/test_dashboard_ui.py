"""Playwright regression tests that drive the app in a real browser."""
import os
import re
import shutil

import pandas as pd
import pytest
from playwright.sync_api import Page, expect

import player_pool

FIXTURE_CSV = os.path.join(os.path.dirname(__file__), "fixtures", "fanduel_sample.csv")


@pytest.fixture
def no_uploaded_pool():
    """Ensures a FanDuel upload from one test doesn't leak into the next, and
    that driving a real upload through the live app doesn't leave test
    fixture rows sitting in the real bundled sample data.

    save_uploaded_csv() refreshes PLAYERS_CSV (data/players.csv) as a side
    effect of every upload, real or test-fixture, so a test that uploads
    FIXTURE_CSV through the app would otherwise overwrite that real file with
    "Test QB One"-style fixture rows.

    UPLOAD_PATH is moved out of the way (this test expects to start from "no
    upload yet") and restored after. PLAYERS_CSV is left in place -- always
    expected to exist -- but backed up and restored, since the test's own
    upload will overwrite it. Snapshotting and restoring rather than just
    deleting matters because these paths can hold real data a person saved
    outside of tests; deleting it unconditionally used to destroy it.
    """
    upload_backup = player_pool.UPLOAD_PATH + ".pretest-backup"
    had_upload = os.path.exists(player_pool.UPLOAD_PATH)
    if had_upload:
        shutil.move(player_pool.UPLOAD_PATH, upload_backup)

    players_backup = player_pool.PLAYERS_CSV + ".pretest-backup"
    had_players_csv = os.path.exists(player_pool.PLAYERS_CSV)
    if had_players_csv:
        shutil.copy2(player_pool.PLAYERS_CSV, players_backup)

    yield

    if os.path.exists(player_pool.UPLOAD_PATH):
        os.remove(player_pool.UPLOAD_PATH)
    if had_upload:
        shutil.move(upload_backup, player_pool.UPLOAD_PATH)

    if had_players_csv:
        shutil.move(players_backup, player_pool.PLAYERS_CSV)


def test_standings_dashboard_loads(page: Page, live_server: str):
    page.goto(live_server + "/")
    expect(page).to_have_title(re.compile("Standings"))
    rows = page.locator(".team-row")
    assert rows.count() == 32  # one row per NFL team


def test_standings_sorted_by_win_pct_desc(page: Page, live_server: str):
    page.goto(live_server + "/")
    win_pct_cells = page.locator(".team-row td:nth-child(5)").all_inner_texts()
    win_pcts = [float(v) for v in win_pct_cells]
    assert win_pcts == sorted(win_pcts, reverse=True)


def test_optimizer_generates_lineup_within_cap(page: Page, live_server: str):
    page.goto(live_server + "/optimizer")
    page.fill("#salary_cap", "60000")
    page.click("#generate-btn")

    expect(page.locator("#lineup-results")).to_be_visible()
    blocks = page.locator(".lineup-block")
    expect(blocks).to_have_count(10)  # 10 distinct lineups, each differing by at least one player

    for i in range(blocks.count()):
        block = blocks.nth(i)
        rows = block.locator(".lineup-row")
        assert rows.count() == 9  # QB, 2RB, 3WR, TE, FLEX, DEF (FanDuel classic)

        total_text = block.locator(".lineup-total-salary").inner_text()
        total_salary = int(re.sub(r"[^\d]", "", total_text))
        assert total_salary <= 60000

    lineups_rosters = [
        frozenset(row.inner_text() for row in blocks.nth(i).locator(".lineup-row td:nth-child(2)").all())
        for i in range(blocks.count())
    ]
    assert len(set(lineups_rosters)) == len(lineups_rosters)  # every lineup is actually distinct


def test_optimizer_rejects_infeasible_cap(page: Page, live_server: str):
    page.goto(live_server + "/optimizer")
    page.fill("#salary_cap", "500")
    page.click("#generate-btn")

    expect(page.locator("#error-message")).to_be_visible()
    expect(page.locator("#lineup-results")).to_have_count(0)


def test_optimizer_rejects_negative_cap(page: Page, live_server: str):
    page.goto(live_server + "/optimizer")
    page.fill("#salary_cap", "-100")
    page.click("#generate-btn")

    expect(page.locator("#error-message")).to_be_visible()


def _unavailable(*args, **kwargs):
    raise RuntimeError("live data not available in this test")


def test_upload_fanduel_csv_switches_player_pool(page: Page, live_server: str, monkeypatch, no_uploaded_pool):
    # The fixture uses fake player names/teams, so skip the real-stats merge
    # and the matchup-eligibility filter -- both would otherwise hit the
    # network, and the fake players could never appear in real ranking data
    # anyway, so the eligibility filter would wipe out this whole fixture.
    monkeypatch.setattr(
        player_pool,
        "get_offense_projections",
        lambda *a, **k: pd.DataFrame(columns=["match_key", "position", "projected_points", "season"]),
    )
    monkeypatch.setattr(player_pool, "build_eligibility", _unavailable)

    page.goto(live_server + "/optimizer")
    expect(page.locator("#data-source")).to_contain_text("sample data")

    page.click("#upload-panel summary")
    page.set_input_files("#fanduel_csv", FIXTURE_CSV)
    page.click("#upload-btn")

    expect(page.locator(".flash-success")).to_be_visible()
    expect(page.locator("#data-source")).to_contain_text("uploaded FanDuel salaries")
    expect(page.locator("#data-source")).to_contain_text("13 players")

    page.fill("#salary_cap", "46000")
    page.click("#generate-btn")

    expect(page.locator("#lineup-results")).to_be_visible()
    first_block = page.locator(".lineup-block").first
    assert first_block.locator(".lineup-row").count() == 9
    names = first_block.locator(".lineup-row td:nth-child(2)").all_inner_texts()
    assert any("Test" in name for name in names)
