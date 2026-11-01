"""Test-wide guards, all autouse: no network, a clean environment, temporary data dirs."""

from __future__ import annotations

import socket
import threading

import pytest

ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "EVO2_BACKEND_URL",
    "EVO2_SCORES_DIR",
    "VUS_PANEL_DATA_DIR",
    "VUS_PANEL_RUNS_DIR",
)

_state = threading.local()
_real_socketpair = socket.socketpair


class NetworkBlocked(RuntimeError):
    """A test tried to reach the network."""


def _socketpair(*args, **kwargs):
    # On Windows, asyncio builds its event loop's self-pipe with socketpair(),
    # which connects two local sockets. That one internal connect is allowed.
    _state.in_socketpair = True
    try:
        return _real_socketpair(*args, **kwargs)
    finally:
        _state.in_socketpair = False


def _blocked(*args, **kwargs):
    raise NetworkBlocked("network access is blocked in tests")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self, address):
        if getattr(_state, "in_socketpair", False):
            return real_connect(self, address)
        raise NetworkBlocked(f"network access is blocked in tests (tried {address!r})")

    def connect_ex(self, address):
        if getattr(_state, "in_socketpair", False):
            return real_connect_ex(self, address)
        raise NetworkBlocked(f"network access is blocked in tests (tried {address!r})")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", _blocked)
    monkeypatch.setattr(socket, "socketpair", _socketpair)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VUS_PANEL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VUS_PANEL_RUNS_DIR", str(tmp_path / "runs"))
