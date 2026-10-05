"""Regression for the Slack reasoning leak (ts 1790772886.423949, bot A0C55QD3DDM).

A bare global ``display.show_reasoning: true`` (set e.g. for CLI use) used to also apply to the
Slack gateway bot because Slack had no per-platform override and was not in the set of platforms
requiring an explicit opt-in. The result: a maintenance-report message to the owner was prefixed
with the model's raw internal reasoning ("Planning backup execution..." in English, mixed with the
German final report). Fixed by requiring an explicit ``display.platforms.slack.show_reasoning:
true`` for Slack, mirroring the existing Mattermost opt-in (gateway/run_turn.py
``_hmwa_prepend_reasoning``). This must be caught at message-composition time in the gateway, not
by a receiver-side text filter.

Unlike an earlier version of this test, the fixture below is NOT a hand-typed/shortened string: it
is the verbatim, unmodified historical incident payload (``fixtures/reasoning_leak_incident_
1790772886.json``, copied byte-for-byte from the communication audit corpus, ts 1790772886.423949)
and the reasoning text is carried through the ACTUAL sender/model extraction path
(``agent.turn_finalizer._last_turn_reasoning``, the same function ``finalize_turn`` uses to
populate ``agent_result["last_reasoning"]``) applied to a messages list shaped exactly like
``agent.chat_completion_helpers.build_assistant_message`` produces it (``role``/``content``/
``reasoning``/``finish_reason``). Nothing sets ``last_reasoning`` by hand.
"""

import json
import re
from pathlib import Path
from typing import Optional

from agent.turn_finalizer import _last_turn_reasoning
from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reasoning_leak_incident_1790772886.json"
_INCIDENT_TS = "1790772886.423949"


def _load_incident() -> dict:
    """Load the unmodified historical incident entry (not re-typed by hand)."""
    entries = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    (entry,) = [e for e in entries if e.get("ts") == _INCIDENT_TS]
    assert entry["app_id"] == "A0C55QD3DDM"  # the affected bot, per the ticket
    return entry


def _split_leaked_text(leaked_text: str) -> tuple[str, str]:
    """Undo the historical leak formatting to recover the two inputs that fed it.

    The buggy Slack message was ``:thought_balloon: *Reasoning:*\\n```<reasoning>```\\n\\n<response>``
    (see the ``gateway.reasoning.block`` template). Splitting on that exact fixed shape — rather
    than re-typing the pieces — keeps the derived reasoning/response text byte-faithful to the
    unmodified original payload.
    """
    match = re.match(r"^:thought_balloon:[^\n]*\n```\n(.*?)\n```\n\n(.*)$", leaked_text, re.DOTALL)
    assert match, "historical fixture text no longer matches the known leaked-message shape"
    return match.group(1), match.group(2)


_INCIDENT = _load_incident()
_HISTORICAL_REASONING, _HISTORICAL_FINAL_RESPONSE = _split_leaked_text(_INCIDENT["text"])

# Sanity checks on the recovered originals: fidelity markers from across the WHOLE original
# message (start, middle, end) so a future accidental truncation of the fixture is caught here
# rather than silently shrinking the regression's coverage.
assert "Planning backup execution" in _HISTORICAL_REASONING
assert "Ich führe die Wartung nur auf dem Hetzner-Server durch" in _HISTORICAL_REASONING
assert "14 zuvor laufenden Container wieder gesund" in _HISTORICAL_FINAL_RESPONSE
assert "Dein Mac wurde nicht verändert" in _HISTORICAL_FINAL_RESPONSE


class _Source:
    def __init__(self, platform):
        self.platform = platform


def _runner_with_show_reasoning(show_reasoning: bool):
    runner = object.__new__(GatewayTurnMixin)
    runner._show_reasoning = show_reasoning
    return runner


def _agent_result_from_real_sender_payload(reasoning: Optional[str], final_response: str) -> dict:
    """Build ``agent_result`` the way ``finalize_turn`` does: run the REAL extraction function
    (``_last_turn_reasoning``) over a messages list shaped like the real model turn output
    (``build_assistant_message``'s ``role``/``content``/``reasoning``/``finish_reason`` dict),
    instead of setting ``last_reasoning`` on a dict by hand."""
    messages = [
        {"role": "user", "content": "Bitte Hetzner-Server aktualisieren und neu starten."},
        {"role": "assistant", "content": final_response, "reasoning": reasoning, "finish_reason": "stop"},
    ]
    return {"last_reasoning": _last_turn_reasoning(messages)}


class TestSlackReasoningLeakRegression:
    def test_extraction_recovers_the_historical_reasoning_unchanged(self):
        """The real sender-path extractor returns exactly the historical reasoning, verbatim."""
        agent_result = _agent_result_from_real_sender_payload(
            _HISTORICAL_REASONING, _HISTORICAL_FINAL_RESPONSE,
        )
        assert agent_result["last_reasoning"] == _HISTORICAL_REASONING

    def test_global_show_reasoning_true_does_not_leak_into_slack(self, tmp_path, monkeypatch):
        """The historical config shape: global show_reasoning=true, no Slack override."""
        import gateway.run as gateway_run

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("display:\n  show_reasoning: true\n", encoding="utf-8")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        agent_result = _agent_result_from_real_sender_payload(
            _HISTORICAL_REASONING, _HISTORICAL_FINAL_RESPONSE,
        )
        runner = _runner_with_show_reasoning(True)
        response = runner._hmwa_prepend_reasoning(
            agent_result, _HISTORICAL_FINAL_RESPONSE, _Source(Platform.SLACK), False,
        )

        assert response == _HISTORICAL_FINAL_RESPONSE
        assert "Planning backup execution" not in response
        assert "Reasoning" not in response

    def test_explicit_slack_opt_in_still_shows_reasoning(self, tmp_path, monkeypatch):
        """An operator who deliberately opts Slack in still gets it (no silent feature removal)."""
        import gateway.run as gateway_run

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "display:\n  platforms:\n    slack:\n      show_reasoning: true\n", encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        agent_result = _agent_result_from_real_sender_payload(
            _HISTORICAL_REASONING, _HISTORICAL_FINAL_RESPONSE,
        )
        runner = _runner_with_show_reasoning(False)
        response = runner._hmwa_prepend_reasoning(
            agent_result, _HISTORICAL_FINAL_RESPONSE, _Source(Platform.SLACK), False,
        )

        assert "Planning backup execution" in response
        assert _HISTORICAL_FINAL_RESPONSE in response

    def test_other_platforms_unaffected_by_slack_opt_in_requirement(self, tmp_path, monkeypatch):
        """Telegram keeps following the bare global switch (#7148) — only Slack/Mattermost tightened."""
        import gateway.run as gateway_run

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("display:\n  show_reasoning: true\n", encoding="utf-8")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        agent_result = _agent_result_from_real_sender_payload(
            _HISTORICAL_REASONING, _HISTORICAL_FINAL_RESPONSE,
        )
        runner = _runner_with_show_reasoning(True)
        response = runner._hmwa_prepend_reasoning(
            agent_result, _HISTORICAL_FINAL_RESPONSE, _Source(Platform.TELEGRAM), False,
        )

        assert "Planning backup execution" in response
