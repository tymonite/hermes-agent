"""Profile delete releases the memory-store handles EVERY loaded provider copy holds in the profile.

Store providers (holographic) are catalog plugins now: each home loads its own copy under
``_hermes_user_memory.<name>__source_<digest>``. The copy holding the doomed profile's SQLite handle
is the one loaded for THAT profile, not the active home's, so a release that resolves the provider
for the active home closes nothing and rmtree fails with WinError 32 on Windows (#88347).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROVIDER_INIT = """
from agent.memory_provider import MemoryProvider


class StoreProvMemoryProvider(MemoryProvider):
    @property
    def name(self):
        return "storeprov"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []
"""

# A process-wide registry of shared connections per DB path, released by directory — the
# ``MemoryStore.release_all_under`` contract ``release_store_handles_under`` drives.
_PROVIDER_STORE = """
import sqlite3
from pathlib import Path


class MemoryStore:
    _shared = {}

    def __init__(self, db_path):
        self.db_path = Path(db_path).resolve()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._shared[str(self.db_path)] = self._conn

    @classmethod
    def release_all_under(cls, directory):
        root = Path(directory).resolve()
        doomed = [key for key in cls._shared if Path(key).is_relative_to(root)]
        for key in doomed:
            cls._shared.pop(key).close()
        return len(doomed)
"""


def _install_store_provider(home: Path) -> None:
    plugin_dir = home / "plugins" / "storeprov"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(_PROVIDER_INIT, encoding="utf-8")
    (plugin_dir / "store.py").write_text(_PROVIDER_STORE, encoding="utf-8")


def _open_store_in(home: Path) -> sqlite3.Connection:
    """Load *home*'s own provider copy (as the agent serving that profile does) and open its DB."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.memory import import_provider_module

    token = set_hermes_home_override(home)
    try:
        store_module = import_provider_module("storeprov", "store")
    finally:
        reset_hermes_home_override(token)
    conn = store_module.MemoryStore(db_path=home / "memory_store.db")._conn
    conn.execute("SELECT 1").fetchone()
    return conn


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    """Active default home A and a named profile B, each with its own copy of a store provider."""
    from hermes_cli.profiles import create_profile

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home_a = tmp_path / ".hermes"
    home_a.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    home_b = create_profile("doomed", no_alias=True)
    for home in (home_a, home_b):
        _install_store_provider(home)
    loaded_before = set(sys.modules)
    yield home_a, home_b
    for name in set(sys.modules) - loaded_before:
        if name.startswith("_hermes_user_memory.storeprov"):
            sys.modules.pop(name, None)


def test_profile_delete_releases_the_doomed_profiles_own_provider_copy(homes, capsys):
    from hermes_cli.profiles import delete_profile

    home_a, home_b = homes
    conn_b = _open_store_in(home_b)
    conn_a = _open_store_in(home_a)
    try:
        with patch("hermes_cli.profiles._cleanup_gateway_service"), \
             patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
            delete_profile("doomed", yes=True)

        assert "Released 1 memory-store connection(s)" in capsys.readouterr().out
        with pytest.raises(sqlite3.ProgrammingError):
            conn_b.execute("SELECT 1")
        assert not home_b.exists()
        conn_a.execute("SELECT 1").fetchone()  # the active home's handle is untouched
    finally:
        conn_a.close()
        conn_b.close()
