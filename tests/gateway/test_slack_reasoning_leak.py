"""Regression for the Slack reasoning leak (incident ts 1790772886.423949, bot A0C55QD3DDM).

A bare global ``display.show_reasoning: true`` (set e.g. for CLI use) used to also apply to the
Slack gateway bot because Slack had no per-platform override and was not in the set of platforms
requiring an explicit opt-in. The result: a message to the owner was prefixed with the model's raw
internal reasoning, mixed in with the final report. Fixed by requiring an explicit
``display.platforms.slack.show_reasoning: true`` for Slack, mirroring the existing Mattermost
opt-in (``gateway/run_turn.py`` ``_hmwa_prepend_reasoning``). This must be caught at
message-composition time in the gateway, not by a receiver-side text filter.

Fixture note: ``fixtures/reasoning_leak_incident_synthetic.json`` is SYNTHETIC. It reproduces only
the structural shape that mattered for the incident (a Slack message whose text is the
reasoning-block template directly followed by the final response, bot id placeholder) — it is NOT
a copy of the real user/model conversation, which is never checked into this public repository.
The reasoning text is still carried through the ACTUAL sender/model extraction path
(``agent.turn_finalizer._last_turn_reasoning``, the same function ``finalize_turn`` uses to
populate ``agent_result["last_reasoning"]``) applied to a messages list shaped exactly like
``agent.chat_completion_helpers.build_assistant_message`` produces it
(``role``/``content``/``reasoning``/``finish_reason``). Nothing sets ``last_reasoning`` by hand.

A separate regression against the REAL, unmodified historical payload exists and was run as part
of this fix's verification, but it lives ONLY in a private, local, non-git evidence directory
outside this repository and is never published (see the task handoff for provenance, command and
result). That private check establishes the real-world claim; this public test establishes the
structural/behavioral claim with safe, synthetic data so the fix is independently reviewable and
re-runnable by anyone without needing access to private data.
"""

import json
import re
from pathlib import Path
from typing import Optional

from agent.turn_finalizer import _last_turn_reasoning
from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reasoning_leak_incident_synthetic.json"


def _load_incident() -> dict:
    (entry,) = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert entry["app_id"] == "A0SYNTH0001"  # synthetic bot id, deliberately distinct from the real one
    return entry


def _split_leaked_text(leaked_text: str) -> tuple[str, str]:
    """Split the synthetic leaked message into the two inputs that fed it (reasoning, response),
    by parsing the same fixed shape the real gateway template produces, rather than hard-coding
    the two halves separately (keeps the test honest about what shape it is checking)."""
    match = re.match(r"^:thought_balloon:[^\n]*\n```\n(.*?)\n```\n\n(.*)$", leaked_text, re.DOTALL)
    assert match, "fixture text no longer matches the known leaked-message shape"
    return match.group(1), match.group(2)


_INCIDENT = _load_incident()
_SYNTH_REASONING, _SYNTH_FINAL_RESPONSE = _split_leaked_text(_INCIDENT["text"])

assert "Planning the requested task" in _SYNTH_REASONING
assert "Task completed successfully" in _SYNTH_FINAL_RESPONSE


class _Source:
    def __init__(self, platform):
        self.platform = platform


def _runner_with_show_reasoning(show_reasoning: bool):
    runner = object.__new__(GatewayTurnMixin)
    runner._show_reasoning = show_reasoning
    return runner


def _agent_result_from_sender_payload(reasoning: Optional[str], final_response: str) -> dict:
    """Build ``agent_result`` the way ``finalize_turn`` does: run the REAL extraction function
    (``_last_turn_reasoning``) over a messages list shaped like the real model turn output
    (``build_assistant_message``'s ``role``/``content``/``reasoning``/``finish_reason`` dict),
    instead of setting ``last_reasoning`` on a dict by hand."""
    messages = [
        {"role": "user", "content": "Please run the requested task."},
        {"role": "assistant", "content": final_response, "reasoning": reasoning, "finish_reason": "stop"},
    ]
    return {"last_reasoning": _last_turn_reasoning(messages)}


class TestSlackReasoningLeakRegression:
    def test_extraction_recovers_the_reasoning_unchanged(self):
        """The real sender-path extractor returns exactly the fixture's reasoning, verbatim."""
        agent_result = _agent_result_from_sender_payload(_SYNTH_REASONING, _SYNTH_FINAL_RESPONSE)
        assert agent_result["last_reasoning"] == _SYNTH_REASONING

    def test_global_show_reasoning_true_does_not_leak_into_slack(self, tmp_path, monkeypatch):
        """The historical config shape: global show_reasoning=true, no Slack override."""
        import gateway.run as gateway_run

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("display:\n  show_reasoning: true\n", encoding="utf-8")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        agent_result = _agent_result_from_sender_payload(_SYNTH_REASONING, _SYNTH_FINAL_RESPONSE)
        runner = _runner_with_show_reasoning(True)
        response = runner._hmwa_prepend_reasoning(
            agent_result, _SYNTH_FINAL_RESPONSE, _Source(Platform.SLACK), False,
        )

        assert response == _SYNTH_FINAL_RESPONSE
        assert "Planning the requested task" not in response
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

        agent_result = _agent_result_from_sender_payload(_SYNTH_REASONING, _SYNTH_FINAL_RESPONSE)
        runner = _runner_with_show_reasoning(False)
        response = runner._hmwa_prepend_reasoning(
            agent_result, _SYNTH_FINAL_RESPONSE, _Source(Platform.SLACK), False,
        )

        assert "Planning the requested task" in response
        assert _SYNTH_FINAL_RESPONSE in response

    def test_other_platforms_unaffected_by_slack_opt_in_requirement(self, tmp_path, monkeypatch):
        """Telegram keeps following the bare global switch (#7148) — only Slack/Mattermost tightened."""
        import gateway.run as gateway_run

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("display:\n  show_reasoning: true\n", encoding="utf-8")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        agent_result = _agent_result_from_sender_payload(_SYNTH_REASONING, _SYNTH_FINAL_RESPONSE)
        runner = _runner_with_show_reasoning(True)
        response = runner._hmwa_prepend_reasoning(
            agent_result, _SYNTH_FINAL_RESPONSE, _Source(Platform.TELEGRAM), False,
        )

        assert "Planning the requested task" in response
