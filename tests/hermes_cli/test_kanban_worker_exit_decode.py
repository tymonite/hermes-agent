"""Worker exit classification must not depend on POSIX-only ``os.WIF*`` helpers.

``_classify_worker_exit`` feeds the dead-worker reclaim: a ``rate_limited``
verdict requeues the card without counting a failure, ``unknown`` counts as a
crash. Windows has neither ``os.WIFEXITED`` nor ``waitpid(-1)``, so both the
decode and the reaper's exit capture have a Windows-independent path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


def _spawn_exit(code: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])  # noqa: S603
    proc.wait()
    return proc




@pytest.mark.skipif(sys.platform == "win32", reason="waitpid retry is POSIX-only")
def test_classify_retries_a_targeted_waitpid_when_the_bulk_sweep_missed_it(monkeypatch):
    """#131204: under a loaded host the bulk ``waitpid(-1)`` sweep in
    ``reap_worker_zombies`` can run a tick (or a fraction of one) before a
    specific child actually zombies, so the registry it fills is still
    empty when the per-board liveness probe already believes that pid is
    dead. A direct child our own process genuinely owns must still be
    reapable via a SECOND, TARGETED ``waitpid`` right there in
    ``_classify_worker_exit`` — the exit code must not be lost just because
    the generic sweep's timing missed it."""
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])  # noqa: S603
    # No .wait()/.poll() here — both call waitpid() themselves and would reap
    # it before the real target gets a chance to. A short, fixed sleep lets
    # the trivially-fast child exit and become a kernel zombie on its own,
    # exactly the state a loaded host's delayed tick observes: dead, but
    # nothing has reaped it yet.
    time.sleep(2.0)
    assert kbd._classify_worker_exit(proc.pid) == ("nonzero_exit", 7)


def test_classify_reports_orphaned_for_a_pid_that_is_not_our_child(monkeypatch):
    """A pid this process never spawned (outlived a gateway restart and was
    reparented away, or simply belongs to someone else) must be told apart
    from an ambiguous timing race: ``waitpid`` raises ``ECHILD``, which is a
    DEFINITIVE answer, not a probe hiccup — the event should say ``orphaned``
    so an operator can tell this crash category from a genuine unknown."""
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})
    # os.getpid() IS a valid, running pid but not our waitpid-able child (it's
    # US) — ECHILD either way, since a process can't wait on itself or on an
    # unrelated pid it never forked.
    assert kbd._classify_worker_exit(os.getpid()) == ("orphaned", None)


@pytest.mark.platforms("windows")
def test_native_windows_reaper_and_decode(monkeypatch):
    """Native Windows, nothing patched: ``_IS_WINDOWS`` selects the Popen-poll
    reaper and the decode runs where ``os.WIFEXITED`` does not exist, so the
    rate-limit sentinel exit is a requeue, not a crash."""
    monkeypatch.setattr(kbd, "_live_worker_procs", {})
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})
    assert not hasattr(os, "WIFEXITED")
    proc = _spawn_exit(kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    kbd._live_worker_procs[proc.pid] = proc
    assert kbd.reap_worker_zombies() == [proc.pid]
    assert kbd._classify_worker_exit(proc.pid) == ("rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE)
