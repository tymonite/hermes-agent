"""Regression for the Slack reasoning leak (ts 1790772886.423949, bot A0C55QD3DDM).

A bare global ``display.show_reasoning: true`` (set e.g. for CLI use) used to also apply to the
Slack gateway bot because Slack had no per-platform override and was not in the set of platforms
requiring an explicit opt-in. The result: a maintenance-report message to the owner was prefixed
with the model's raw internal reasoning ("Planning backup execution..." in English, mixed with the
German final report). Fixed by requiring an explicit ``display.platforms.slack.show_reasoning:
true`` for Slack, mirroring the existing Mattermost opt-in (gateway/run_turn.py
``_hmwa_prepend_reasoning``). This must be caught at message-composition time in the gateway, not
by a receiver-side text filter.
"""

from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin


# The exact internal reasoning text from the historical incident (corpus.json, ts
# 1790772886.423949), and the final report that should reach Slack on its own.
_HISTORICAL_REASONING = (
    "**Planning backup execution**\n\n"
    "I need to perform a quick backup, possibly using pg dumps and securing it remotely. "
    "I'll use a script with stdin, SSH, and Python, but I want to avoid using heredoc for "
    "script creation. Instead, I'll write a local bash script and execute it through SSH."
)
_HISTORICAL_FINAL_RESPONSE = (
    "*Hetzner-Server aktualisiert und erfolgreich neu gestartet.*\n\n"
    "- Alle *41 Paketupdates installiert*; keine weiteren Paketupdates offen.\n"
    "- Neustart bestätigt; keine weitere Neustartanforderung und keine fehlgeschlagenen Systemdienste."
)


class _Source:
    def __init__(self, platform):
        self.platform = platform


def _runner_with_show_reasoning(show_reasoning: bool):
    runner = object.__new__(GatewayTurnMixin)
    runner._show_reasoning = show_reasoning
    return runner


def _agent_result():
    return {"last_reasoning": _HISTORICAL_REASONING}


class TestSlackReasoningLeakRegression:
    def test_global_show_reasoning_true_does_not_leak_into_slack(self, tmp_path, monkeypatch):
        """The historical config shape: global show_reasoning=true, no Slack override."""
        import gateway.run as gateway_run

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("display:\n  show_reasoning: true\n", encoding="utf-8")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _runner_with_show_reasoning(True)
        response = runner._hmwa_prepend_reasoning(
            _agent_result(), _HISTORICAL_FINAL_RESPONSE, _Source(Platform.SLACK), False,
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

        runner = _runner_with_show_reasoning(False)
        response = runner._hmwa_prepend_reasoning(
            _agent_result(), _HISTORICAL_FINAL_RESPONSE, _Source(Platform.SLACK), False,
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

        runner = _runner_with_show_reasoning(True)
        response = runner._hmwa_prepend_reasoning(
            _agent_result(), _HISTORICAL_FINAL_RESPONSE, _Source(Platform.TELEGRAM), False,
        )

        assert "Planning backup execution" in response
