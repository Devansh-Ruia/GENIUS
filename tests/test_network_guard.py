"""The autouse guards in conftest.py are active in every test."""

from __future__ import annotations

import os
import socket

import pytest


def test_outbound_connections_are_blocked():
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    sock = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="network access is blocked"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()


def test_dns_lookups_are_blocked():
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.getaddrinfo("example.invalid", 80)


def test_socketpair_still_works_for_event_loops():
    left, right = socket.socketpair()
    try:
        left.sendall(b"x")
        assert right.recv(1) == b"x"
    finally:
        left.close()
        right.close()


def test_secrets_are_cleared_and_data_dirs_are_temporary(tmp_path):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "EVO2_BACKEND_URL"):
        assert name not in os.environ
    assert os.environ["VUS_PANEL_DATA_DIR"] == str(tmp_path / "data")
    assert os.environ["VUS_PANEL_RUNS_DIR"] == str(tmp_path / "runs")
