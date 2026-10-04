"""A transiently unreadable process probe must never be mistaken for a dead worker.

Incident (2026-10-04): the kanban dispatcher declared every running worker on
the host dead in the same tick, four times in one day (27 false crashes, 9
cards auto-blocked). Root cause: ``_pid_alive``'s secondary zombie probe
(``ps``) and ``_pid_recycled``'s fingerprint re-read both treated "I could not
read this" the same as "confirmed dead/foreign" — so a momentary ``ps``
hiccup or an unreadable ``/proc``/``get_process_start_time`` answer downgraded
every live worker to "dead" at once, spawning a duplicate worker on top of
each one still running.

Fix: fail-dead only on a CONFIRMED signal (``kill(0)`` ESRCH, or a positively
read zombie/foreign-fingerprint state). An unreadable secondary probe answers
"unknown", and unknown means "still ours" — never "dead". A circuit breaker
on ``_reclaim_dead_workers`` additionally treats >= 3 same-tick "dead, no exit
code harvested" tasks as a suspected systemic probe failure and defers
reclaiming them for a confirmation window instead of respawning duplicates.
"""

from __future__ import annotations

import os
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kbd._mass_crash_first_seen.clear()
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()
        kbd._mass_crash_first_seen.clear()


def _claimed_running(conn, *, pid: int, started_at=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    if started_at is not None:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?", (started_at, tid))
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?", (old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


# ---------------------------------------------------------------------------
# _pid_alive: an unreadable secondary zombie probe must not override a
# kill(0)-confirmed "alive" answer.
# ---------------------------------------------------------------------------

def test_pid_alive_survives_a_failing_secondary_probe(monkeypatch):
    """``ps``/``/proc`` erroring out (sandboxed host, transient resource
    exhaustion) on a definitely-live PID must not flip it to dead."""
    import sys

    live_pid = os.getpid()
    if sys.platform == "darwin":
        class _FakeFailedPs:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(kbd.subprocess, "run", lambda *a, **k: _FakeFailedPs())
        assert kbd._pid_alive(live_pid) is True
    elif sys.platform == "linux":
        import builtins

        real_open = builtins.open

        def _raising_open(path, *a, **k):
            if str(path).startswith("/proc/"):
                raise PermissionError("simulated unreadable /proc entry")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", _raising_open)
        assert kbd._pid_alive(live_pid) is True
    else:
        pytest.skip("probe simulated only for darwin/linux")


def test_pid_alive_still_catches_a_confirmed_zombie(monkeypatch):
    """A positively read zombie state is still dead — the fix only removes
    the fail-dead-on-unreadable behaviour, not real zombie detection."""
    import sys

    if sys.platform != "darwin":
        pytest.skip("ps stat= zombie probe is darwin-specific here")

    class _FakeZombiePs:
        returncode = 0
        stdout = "Z\n"

    monkeypatch.setattr(kbd.subprocess, "run", lambda *a, **k: _FakeZombiePs())
    assert kbd._pid_alive(os.getpid()) is False


def test_pid_alive_confirmed_dead_pid_stays_dead():
    """A PID that never existed is still reported dead (kill(0) ESRCH) —
    the fix changes nothing about genuinely confirmed deaths."""
    # Spawn and reap a child so its PID is guaranteed gone.
    pid = os.fork() if hasattr(os, "fork") else None
    if pid == 0:
        os._exit(0)  # pragma: no cover - child path
    if pid:
        os.waitpid(pid, 0)
        assert kbd._pid_alive(pid) is False
    else:
        pytest.skip("os.fork unavailable on this platform")


# ---------------------------------------------------------------------------
# _pid_recycled: an unreadable fingerprint re-read must not be mistaken for
# a confirmed PID recycle.
# ---------------------------------------------------------------------------

def test_pid_recycled_false_when_legacy_start_time_unreadable(monkeypatch):
    """Old-style integer fingerprint, re-read via ``get_process_start_time``:
    a transient ``None`` answer must NOT count as "recycled"."""
    import gateway.status as status

    monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)
    assert kbd._pid_recycled(os.getpid(), 12345) is False


def test_pid_recycled_false_when_new_style_fingerprint_unreadable(monkeypatch):
    """New-style ``"<epoch>|<start>"`` fingerprint: an unreadable re-read
    (``_process_fingerprint`` returns ``None``) must NOT count as "recycled"."""
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    assert kbd._pid_recycled(os.getpid(), "deadbeef-boot:1|12345") is False


def test_pid_recycled_still_true_for_a_genuinely_different_process(monkeypatch):
    """A successfully read, genuinely different fingerprint is still a
    confirmed recycle — the fix only changes the unreadable case."""
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: "deadbeef-boot:1|999999")
    assert kbd._pid_recycled(os.getpid(), "deadbeef-boot:1|12345") is True


# ---------------------------------------------------------------------------
# Mass-crash circuit breaker on _reclaim_dead_workers / detect_crashed_workers.
# ---------------------------------------------------------------------------

def test_mass_unconfirmed_death_is_deferred_not_reclaimed(board, monkeypatch):
    """>= 3 tasks declared dead in the same tick, none with a harvested exit
    code, is treated as a suspected systemic probe failure: none are
    reclaimed immediately, and a warning is logged."""
    conn = board
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    tids = [_claimed_running(conn, pid=90000 + i) for i in range(3)]

    warnings = []
    monkeypatch.setattr(kb._log, "warning", lambda *a, **k: warnings.append((a, k)))

    crashed = kbd.detect_crashed_workers(conn)
    assert crashed == []
    assert warnings, "expected a warning about the suspected mass probe failure"
    for tid in tids:
        assert kb.get_task(conn, tid).status == "running"


def test_mass_unconfirmed_death_reclaimed_after_confirmation_window(board, monkeypatch):
    """The SAME cohort, still dead after the confirmation window has
    elapsed, is reclaimed — the breaker defers once, it does not livelock."""
    conn = board
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    tids = [_claimed_running(conn, pid=91000 + i) for i in range(3)]

    assert kbd.detect_crashed_workers(conn) == []

    future = time.time() + kbd._MASS_CRASH_DEFER_SECONDS + 1
    monkeypatch.setattr(kbd.time, "time", lambda: future)
    crashed = kbd.detect_crashed_workers(conn)
    assert sorted(crashed) == sorted(tids)
    for tid in tids:
        # Reclaimed (released from the dead worker) — the pre-existing
        # identical-fingerprint systemic-crash accounting (_account_crashes)
        # then blocks them for an operator, independent of this breaker.
        assert kb.get_task(conn, tid).status in ("ready", "blocked")


def test_below_threshold_dead_cohort_reclaims_immediately(board, monkeypatch):
    """Only 2 simultaneous unconfirmed deaths (below the mass threshold of 3)
    is ordinary behaviour: reclaimed right away, no deferral."""
    conn = board
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    tids = [_claimed_running(conn, pid=92000 + i) for i in range(2)]

    crashed = kbd.detect_crashed_workers(conn)
    assert sorted(crashed) == sorted(tids)
    for tid in tids:
        assert kb.get_task(conn, tid).status == "ready"


def test_mass_cohort_member_coming_back_alive_is_not_reclaimed_later(board, monkeypatch):
    """A task in a deferred mass cohort whose worker turns out to still be
    alive on the next tick is simply dropped from the cohort, never
    reclaimed via stale per-task state."""
    conn = board
    alive = {"flag": False}
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: alive["flag"])
    tids = [_claimed_running(conn, pid=93000 + i) for i in range(3)]

    assert kbd.detect_crashed_workers(conn) == []

    alive["flag"] = True  # the probe glitch cleared
    future = time.time() + kbd._MASS_CRASH_DEFER_SECONDS + 1
    monkeypatch.setattr(kbd.time, "time", lambda: future)
    assert kbd.detect_crashed_workers(conn) == []
    for tid in tids:
        assert kb.get_task(conn, tid).status == "running"
    assert not kbd._mass_crash_first_seen
