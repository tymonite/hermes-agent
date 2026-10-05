"""Fixture-E2E for the ``reopen-done`` recovery path (DRE-449/DRE-434).

Scenario mirrored from the real incident: a task completed and bound its
``completion_contract`` to PR39, but the actual accepted fix is a *later*,
still-open PR40 on the same identifier/assignee/history. Neither
``promote_task`` (todo/blocked only) nor ``reopen_review_task`` (review only)
nor ``edit_task`` (never touches status/completion_contract) can reach a
``done`` task, so this exercises the dedicated recovery op end to end through
the real public CLI (``hermes kanban reopen-done`` via ``build_parser`` +
``kanban_command``) as well as the underlying db function directly for the
negative/atomicity cases.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


_PR39 = "https://github.com/dree-projects/agent-ops/pull/39"
_PR40 = "https://github.com/dree-projects/agent-ops/pull/40"


def _done_with_contract(conn, *, contract=_PR39, assignee="claude-dev"):
    """Build the #DRE-449 scenario: a task closed 'done' bound to one PR.

    Direct-SQL terminal state (project convention, see ``test_kanban_promote.py``):
    ``complete_task`` would run the real GitHub PR-acceptance network check for a
    non-local-only contract, which this fixture must not depend on.
    """
    tid = kb.create_task(conn, title="server audit rework", assignee=assignee)
    conn.execute(
        "UPDATE tasks SET status='done', completion_contract=?, completed_at=?, "
        "result=? WHERE id=?",
        (contract, int(time.time()), "shipped PR39", tid),
    )
    row = conn.execute("SELECT status, completion_contract FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "done" and row["completion_contract"] == contract
    return tid


# ---------------------------------------------------------------------------
# db.reopen_done_task_for_rework — direct unit coverage
# ---------------------------------------------------------------------------


def test_reopen_done_rebinds_to_corrected_pr_and_preserves_identity(conn):
    tid = _done_with_contract(conn)
    assignee_before = kb.get_task(conn, tid).assignee

    ok, err = kb.reopen_done_task_for_rework(
        conn, tid, expected_status="done", expected_contract=_PR39,
        new_contract=_PR40, actor="cto", reason="PR40 is the real accepted fix",
    )
    assert ok and err is None

    task = kb.get_task(conn, tid)
    assert task.status in ("ready", "todo")  # landing status after parents
    assert task.completion_contract == _PR40
    assert task.completed_at is None
    assert task.assignee == assignee_before  # same identifier/assignee, not a new card
    assert task.id == tid

    events = [e for e in kb.list_events(conn, tid) if e.kind == "reopened_for_rework"]
    assert len(events) == 1
    payload = events[0].payload
    assert payload["prior_status"] == "done" and payload["new_status"] == task.status
    assert payload["prior_contract"] == _PR39 and payload["new_contract"] == _PR40
    assert payload["reason"] == "PR40 is the real accepted fix"

    # The rework audit event is additive — nothing else on the event log is rewritten.
    assert len(events) == 1


def test_reopen_done_refuses_non_done_expected_status(conn):
    tid = _done_with_contract(conn)
    ok, err = kb.reopen_done_task_for_rework(
        conn, tid, expected_status="review", expected_contract=_PR39,
        new_contract=_PR40, actor="cto",
    )
    assert not ok and "done" in err
    assert kb.get_task(conn, tid).status == "done"  # no partial mutation


def test_reopen_done_refuses_actually_non_done_task(conn):
    tid = kb.create_task(conn, title="still open", completion_contract=_PR39)
    ok, err = kb.reopen_done_task_for_rework(
        conn, tid, expected_status="done", expected_contract=_PR39,
        new_contract=_PR40, actor="cto",
    )
    assert not ok and "not 'done'" in err
    assert kb.get_task(conn, tid).status == "ready"


def test_reopen_done_refuses_stale_contract_expectation(conn):
    tid = _done_with_contract(conn, contract=_PR39)
    # Caller's belief about the current contract is wrong (e.g. stale read).
    ok, err = kb.reopen_done_task_for_rework(
        conn, tid, expected_status="done",
        expected_contract="https://github.com/dree-projects/agent-ops/pull/38",
        new_contract=_PR40, actor="cto",
    )
    assert not ok and "stale expectation" in err
    task = kb.get_task(conn, tid)
    assert task.status == "done" and task.completion_contract == _PR39  # untouched


def test_reopen_done_refuses_noop_rebind(conn):
    tid = _done_with_contract(conn)
    ok, err = kb.reopen_done_task_for_rework(
        conn, tid, expected_status="done", expected_contract=_PR39,
        new_contract=_PR39, actor="cto",
    )
    assert not ok and "identical" in err
    assert kb.get_task(conn, tid).status == "done"


def test_reopen_done_refuses_silent_local_only_downgrade(conn):
    tid = _done_with_contract(conn)
    ok, err = kb.reopen_done_task_for_rework(
        conn, tid, expected_status="done", expected_contract=_PR39,
        new_contract="local-only", actor="cto",
    )
    assert not ok and "local-only" in err
    assert kb.get_task(conn, tid).status == "done"


def test_reopen_done_refuses_malformed_new_contract(conn):
    tid = _done_with_contract(conn)
    ok, err = kb.reopen_done_task_for_rework(
        conn, tid, expected_status="done", expected_contract=_PR39,
        new_contract="not a repo or pr url", actor="cto",
    )
    assert not ok and err  # validate_contract's ValueError message
    assert kb.get_task(conn, tid).status == "done"


def test_reopen_done_refuses_while_a_worker_is_actively_claimed(conn):
    tid = _done_with_contract(conn)
    # Simulate a live claim surviving on the row (defensive: done tasks normally
    # have none, but the guard must still hold if one is ever present).
    conn.execute(
        "UPDATE tasks SET claim_lock=?, claim_expires=?, worker_pid=? WHERE id=?",
        ("someone:1", 9999999999, 12345, tid),
    )
    ok, err = kb.reopen_done_task_for_rework(
        conn, tid, expected_status="done", expected_contract=_PR39,
        new_contract=_PR40, actor="cto",
    )
    assert not ok and "active run/claim" in err


def test_reopen_done_refuses_unknown_task(conn):
    ok, err = kb.reopen_done_task_for_rework(
        conn, "t_doesnotexist", expected_status="done", expected_contract=_PR39,
        new_contract=_PR40, actor="cto",
    )
    assert not ok and "not found" in err


def test_reopen_done_lands_in_todo_when_a_parent_is_still_open(conn):
    parent = kb.create_task(conn, title="parent still running")
    tid = kb.create_task(conn, title="child rework", parents=[parent],
                          completion_contract=_PR39)
    # Force the child to 'done' despite an open parent (the exact premature-close
    # shape this recovery op exists for).
    conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))

    ok, err = kb.reopen_done_task_for_rework(
        conn, tid, expected_status="done", expected_contract=_PR39,
        new_contract=_PR40, actor="cto",
    )
    assert ok and err is None
    assert kb.get_task(conn, tid).status == "todo"  # re-gated, not a bare 'ready'


# ---------------------------------------------------------------------------
# Real public CLI: `hermes kanban reopen-done` via build_parser + kanban_command
# ---------------------------------------------------------------------------


def _parse(argv):
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser.parse_args(argv)


def test_cli_reopen_done_rebinds_and_reads_back(kanban_home, capsys):
    with kbc.connect() as conn:
        tid = _done_with_contract(conn)

    args = _parse([
        "kanban", "reopen-done", tid,
        "--expected-contract", _PR39, "--new-contract", _PR40,
        "--reason", "PR40 is the real accepted fix",
    ])
    rc = kc.kanban_command(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert tid in out and _PR40 in out

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status in ("ready", "todo")
    assert task.completion_contract == _PR40
    assert task.completed_at is None


def test_cli_reopen_done_refuses_stale_contract_with_clear_message(kanban_home, capsys):
    with kbc.connect() as conn:
        tid = _done_with_contract(conn)

    args = _parse([
        "kanban", "reopen-done", tid,
        "--expected-contract", "https://github.com/dree-projects/agent-ops/pull/999",
        "--new-contract", _PR40,
    ])
    rc = kc.kanban_command(args)
    assert rc != 0
    err = capsys.readouterr().err
    assert "stale expectation" in err

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "done"  # failed call, no mutation


def test_cli_reopen_done_denied_from_a_worker_task_context(kanban_home, monkeypatch, capsys):
    """Orchestrator-only, like `unblock`: a running worker must hand off, not self-reopen."""
    with kbc.connect() as conn:
        tid = _done_with_contract(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_someotherrunningtask")

    args = _parse([
        "kanban", "reopen-done", tid,
        "--expected-contract", _PR39, "--new-contract", _PR40,
    ])
    rc = kc.kanban_command(args)
    assert rc != 0
    assert "orchestrator-only" in capsys.readouterr().err

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "done"
