"""Spins up the real Flask app in a background thread so Playwright can drive it
like an actual browser user, instead of testing against Flask's test client."""
import socket
import threading

import pytest
from werkzeug.serving import make_server

from app import app as flask_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server():
    port = _free_port()
    server = make_server("127.0.0.1", port, flask_app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


@pytest.fixture(autouse=True)
def no_live_nfl_database(tmp_path, monkeypatch):
    """Keep tests off the real NFL Assistant database, whose injury report
    changes daily. Tests that want one point NFL_DATABASE_URL at a fixture."""
    import nfl_data

    monkeypatch.setenv("NFL_DATABASE_URL", f"sqlite:///{(tmp_path / 'no-nfl.db').as_posix()}")
    nfl_data.reset()
    yield
    nfl_data.reset()
