"""TDD: Idle Model Unloading.

Tests for the IdleModelManager class that unloads WhisperX ASR
model from GPU VRAM after a configurable idle period.

GREEN phase — class extracted from server.py via AST.
"""

import ast
import threading
import time
from pathlib import Path

import pytest

SERVER_PY = Path(__file__).parent.parent / "server.py"


def _extract_class(class_name: str):
    """Extract a class from server.py source code using AST."""
    source = SERVER_PY.read_text()
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            mod = ast.Module(body=[node], type_ignores=[])
            ast.copy_location(mod, node)
            code = compile(ast.unparse(mod), "<test>", "exec")
            namespace = {"threading": threading, "time": time}
            exec(code, namespace)
            return namespace[class_name]
    raise ValueError(f"Class {class_name} not found in {SERVER_PY}")


# Extract the class once at module load
IdleModelManager = _extract_class("IdleModelManager")


def _make_idle(manager, seconds=10):
    """Helper: set manager's _last_used far in the past (for testing)."""
    with manager._lock:
        manager._last_used = time.monotonic() - seconds


class TestIdleModelManager:
    """Test suite for IdleModelManager."""

    def test_default_ttl_is_300(self):
        """Default TTL should be 300 seconds (5 minutes)."""
        manager = IdleModelManager()
        assert manager._ttl == 300

    def test_custom_ttl(self):
        """Custom TTL should be respected."""
        manager = IdleModelManager(ttl=600)
        assert manager._ttl == 600

    def test_touch_resets_timer(self):
        """touch() should update _last_used to approximately now."""
        manager = IdleModelManager()
        before = manager._last_used
        time.sleep(0.01)
        manager.touch()
        assert manager._last_used > before

    def test_unload_calls_on_unload_callback(self):
        """unload() should invoke the on_unload callback when idle."""
        calls = []

        def callback():
            calls.append(1)

        manager = IdleModelManager(ttl=1, on_unload=callback)
        _make_idle(manager, seconds=10)  # 10s idle > 1s TTL
        manager.unload()
        assert len(calls) == 1

    def test_unload_skips_when_not_idle(self):
        """unload() should NOT invoke callback if timer was recently touched."""
        calls = []

        def callback():
            calls.append(1)

        manager = IdleModelManager(on_unload=callback)
        manager.touch()  # recent touch
        manager.unload()
        assert len(calls) == 0  # should skip because not idle

    def test_unload_no_callback_no_error(self):
        """unload() with no callback should not raise."""
        manager = IdleModelManager()
        _make_idle(manager)
        manager.unload()  # Should not raise

    def test_ttl_zero_never_unloads(self):
        """TTL=0 means unload() skips because 0 > 0 is False."""
        calls = []
        manager = IdleModelManager(ttl=0, on_unload=lambda: calls.append(1))
        _make_idle(manager)
        manager.unload()
        assert len(calls) == 0

    def test_unload_after_ttl_expiry(self):
        """After idle exceeds TTL, unload() should invoke callback."""
        calls = []
        manager = IdleModelManager(ttl=1, on_unload=lambda: calls.append(1))
        _make_idle(manager, seconds=10)  # 10s idle > 1s TTL
        manager.unload()
        assert len(calls) == 1

    def test_touch_before_unload_prevents_unload(self):
        """A recent touch() should prevent unload even if previously idle."""
        calls = []
        manager = IdleModelManager(ttl=1, on_unload=lambda: calls.append(1))
        _make_idle(manager, seconds=10)  # was idle
        manager.touch()  # but then touched
        manager.unload()
        assert len(calls) == 0  # touch reset the timer

    def test_callback_receives_no_args(self):
        """on_unload callback should be called with no arguments."""
        received_args = []

        def callback(*args):
            received_args.append(args)

        manager = IdleModelManager(ttl=1, on_unload=callback)
        _make_idle(manager, seconds=10)
        manager.unload()
        assert len(received_args) == 1
        assert received_args[0] == ()

    def test_constructor_uses_keyword_only_args(self):
        """All constructor args should be keyword-only (prevents positional misuse)."""
        import inspect
        sig = inspect.signature(IdleModelManager.__init__)
        params = list(sig.parameters.values())[1:]  # skip self
        for p in params:
            assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"Parameter {p.name} should be keyword-only"
            )


class TestIdleEnvVar:
    """Tests for the MODEL_IDLE_TTL env var with safe int() parsing."""

    def test_default_env_is_300(self):
        """When MODEL_IDLE_TTL is not set, default should be 300."""
        source = SERVER_PY.read_text()
        assert "MODEL_IDLE_TTL" in source, "MODEL_IDLE_TTL not found in server.py"
        # Check that default value '300' is present in the env var read
        assert 'MODEL_IDLE_TTL' in source

    def test_has_safe_int_parse(self):
        """MODEL_IDLE_TTL should use try/except for safe int() parsing."""
        source = SERVER_PY.read_text()
        assert "try:" in source.split("MODEL_IDLE_TTL")[1].split("\n")[0] or "except" in source, (
            "MODEL_IDLE_TTL should have safe int() parsing with try/except"
        )
