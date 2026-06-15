# Plan: Idle Model Unloading

## Summary

Add an `IdleModelManager` that monitors the WhisperX ASR model (`_whisperx_model`) and unloads it from GPU VRAM after a configurable idle period (~3GB freed). The model reloads lazily on the next `/align` request.

## Technical Context

- **Demucs**: runs as subprocess, exits after each request — no VRAM issue.
- **WhisperX ASR** (`_whisperx_model`): singleton global, lives forever in GPU memory (~3GB). **Target of this change.**
- **CREPE**: called per-request via `torchcrepe.predict()` (internal caching, not persistent).
- **wav2vec2 aligners**: already have LRU eviction capped at `MAX_WHISPERX_ALIGNERS`.
- Server uses `expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF` to reduce fragmentation.
- `_get_whisperx_model()` at `server.py:399` is the lazy-load gateway — always checks `if _whisperx_model is None`.

## Design

### `IdleModelManager` class

```text
IdleModelManager:
  - _model_ref: reference to module-level _whisperx_model (via closure/global)
  - _last_used: float timestamp (time.monotonic)
  - _ttl: int seconds (from env MODEL_IDLE_TTL, default 300)
  - _lock: threading.Lock

  Methods:
    touch()       — updates _last_used
    _check_loop() — daemon thread: every 60s, if idle > TTL, unload
    unload()      — set _whisperx_model = None, torch.cuda.empty_cache(), update warmup_state
```

### Threading

- Background daemon thread created in `_startup_event()` alongside cache cleanup.
- Check interval: 60 seconds (hardcoded, not configurable).
- `touch()` called from within `_get_whisperx_model()` (at return) and from `/align` endpoint scope.

### States added to `warmup_state`

- `"whisperx"`: `"unloaded"` — new value, set by `IdleModelManager.unload()`.
- On re-load, `_mark_lazy_loaded("whisperx")` transitions back to `"ready"`.
- `"unloaded"` is sanitized to `"unloaded"` (safe for /health — no internal detail leaked).

### Config

| Env var | Default | Description |
|---------|---------|-------------|
| `MODEL_IDLE_TTL` | `300` | Seconds of inactivity before unload. `0` disables unloading. |

## Files

| File | Change |
|------|--------|
| `server.py` | Add `IdleModelManager` class, daemon thread in `_startup_event`, `touch()` calls in `_get_whisperx_model`, `"unloaded"` in warmup_state |
| `tests/test_idle_unload.py` | New test file using AST extraction pattern (same as `test_cache_cleanup.py`) |
| `docker-compose.yml` | Add `MODEL_IDLE_TTL=300` to environment |
| `Dockerfile` | Add `MODEL_IDLE_TTL=300` to `ENV` block |

## Test Plan

Test file: `tests/test_idle_unload.py`

Using the same AST-extraction pattern from `test_cache_cleanup.py` to avoid the torch/whisperx import chain.

### Scenarios

1. **`MODEL_IDLE_TTL=0` disables unloading** — `IdleModelManager` with TTL=0 never calls `unload()`.
2. **touch defers unload** — after `touch()`, the idle timer resets.
3. **unload sets `_whisperx_model = None`** — verify the global is set to None.
4. **unload calls `torch.cuda.empty_cache()`** — mock/patch to verify call.
5. **unload sets warmup_state["whisperx"] = "unloaded"** — state transition.
6. **reload after unload** — `_get_whisperx_model()` reinitializes when `_whisperx_model is None`.
7. **health reflects "unloaded"** — `"whisperx": "unloaded"` in `/health` response.
8. **Default TTL is 300s** — no env var set.

## Tasks

### 1. `IdleModelManager` class — `server.py`

- Add `import threading` (already present), `time` (already present).
- Read `MODEL_IDLE_TTL` env var at module level.
- Define `IdleModelManager` class with `__init__`, `touch`, `unload`, `_check_loop`.
- Thread safety: use `_lock` around `_last_used` reads/writes.

### 2. Add "unloaded" state support

- Add `"unloaded"` to the set of possible `warmup_state` values.
- Ensure `_sanitize` in `/health` passes `"unloaded"` through (it starts with no prefix, so it passes).
- Ensure `_mark_lazy_loaded` transitions "unloaded" → "ready" correctly (it will: current check is `if current in ("ready", "downloading"): return`, and `"unloaded"` is neither, so it will call `_set_warmup_state("whisperx", "ready")`).

### 3. Integrate into startup

- In `_startup_event`, after cache cleanup thread start, create an `IdleModelManager` instance and start its `_check_loop` daemon thread.
- Store manager on `app.state.idle_manager`.

### 4. Integrate touch into `_get_whisperx_model`

- At the end of `_get_whisperx_model()`, call manager's `touch()`.
- Need to pass manager reference (or use a module-level global).

### 5. Integrate touch into `/align` endpoint

- Also call `touch()` at the start of `/align` to refresh the timer on each request (belt-and-suspenders with the call inside `_get_whisperx_model`).

### 6. Integration test

- Write tests in `tests/test_idle_unload.py` using AST extraction.

### 7. Docker config

- Add `MODEL_IDLE_TTL=300` to `docker-compose.yml` environment section.
- Add `MODEL_IDLE_TTL=300` to `Dockerfile` ENV block.

## Implementation approach

TDD (red-green-refactor):

1. Write `tests/test_idle_unload.py` with all test scenarios
2. Implement `IdleModelManager` in `server.py`
3. Wire into `_get_whisperx_model`, `_startup_event`, `/align`, `/health`
4. Update Docker files
5. Run tests: `python -m pytest tests/test_idle_unload.py -v`
