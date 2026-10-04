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
    kbd._global_unknown_cohort.clear()
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()
        kbd._mass_crash_first_seen.clear()
        kbd._global_unknown_cohort.clear()


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


def test_pid_recycled_false_for_macos_boottime_drift_same_process(monkeypatch):
    """A ~1s start-time drift on the SAME process (macOS kern.boottime shift
    across a sleep/wake or NTP correction, #117505) must NOT read as a
    recycle: the numeric start-time half of the fingerprint is compared with
    ``START_TIME_DRIFT_TOLERANCE`` slack, not exact string equality. This is
    the root cause behind workers being falsely declared dead: on macOS
    ``current_instantiation_epoch()`` is always empty (Linux-only), so the
    fingerprint reduces to just this drift-prone start-time reading."""
    # Centisecond units (×100): 50 centiseconds = 0.5s drift, well inside the
    # 200-centisecond (~2s) tolerance.
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: "|500050")
    assert kbd._pid_recycled(os.getpid(), "|500000") is False


def test_pid_recycled_true_when_drift_exceeds_tolerance(monkeypatch):
    """A start-time difference well beyond the drift tolerance is still a
    genuine recycle, not swallowed by the new slack."""
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: "|999999")
    assert kbd._pid_recycled(os.getpid(), "|500000") is True


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


def test_mass_crash_threshold_counts_across_boards(tmp_path, monkeypatch):
    """Two boards with only 1-2 unconfirmed deaths each (below the per-board
    threshold of 3) must still be recognised as ONE systemic probe-failure
    cohort when the embedded gateway dispatches both in the same tick:
    combined they cross ``_MASS_CRASH_MIN_COUNT``. The dispatcher visits
    boards sequentially within one tick, so the board visited FIRST can't
    see a sibling board's contribution yet (a one-tick lag is the accepted
    trade-off, not a silent miss) — but the SAME-tick sibling visited right
    after it does, and from the next tick onward (if the cohort is still
    dead) every board in it defers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kbd._mass_crash_first_seen.clear()
    kbd._global_unknown_cohort.clear()
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    conn_a = kbc.connect(tmp_path / "board-a.db")
    conn_b = kbc.connect(tmp_path / "board-b.db")
    try:
        tids_a = [_claimed_running(conn_a, pid=94000 + i) for i in range(2)]
        tids_b = [_claimed_running(conn_b, pid=95000 + i) for i in range(1)]

        # Tick 1: board A is visited first with only 2 dead of its own (below
        # threshold) and reclaims immediately — exactly like the pre-existing
        # single-board behaviour, since nothing cross-board is visible yet.
        crashed_a = kbd.detect_crashed_workers(conn_a, board="board-a")
        assert sorted(crashed_a) == sorted(tids_a)
        for tid in tids_a:
            assert kb.get_task(conn_a, tid).status == "ready"

        # Board B, visited right after in the SAME tick, now folds in board
        # A's just-reported cohort (still held, not yet pruned) and crosses
        # the threshold (2 + 1 = 3): it defers instead of reclaiming.
        crashed_b = kbd.detect_crashed_workers(conn_b, board="board-b")
        assert crashed_b == []
        for tid in tids_b:
            assert kb.get_task(conn_b, tid).status == "running"

        # After the confirmation window board B's worker is STILL dead -> reclaimed.
        future = time.time() + kbd._MASS_CRASH_DEFER_SECONDS + 1
        monkeypatch.setattr(kbd.time, "time", lambda: future)
        crashed_b = kbd.detect_crashed_workers(conn_b, board="board-b")
        assert sorted(crashed_b) == sorted(tids_b)
    finally:
        conn_a.close()
        conn_b.close()
        kbd._mass_crash_first_seen.clear()
        kbd._global_unknown_cohort.clear()
