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
  - _in_flight: int counter (active request tracking)
  - _last_used: float timestamp (time.monotonic)
  - _ttl: int seconds (from env MODEL_IDLE_TTL, default 300)
  - _lock: threading.Lock

  Methods:
    touch()       — updates _last_used (reset idle timer)
    acquire()     — touch + increment in_flight (atomic, prevents unload during active request)
    release()     — decrement in_flight (allows unload when idle)
    unload()      — check _in_flight > 0 (skip if active), set _whisperx_model = None,
                    torch.cuda.empty_cache(), update warmup_state
```

### Threading

- Background daemon thread created in `_startup_event()` alongside cache cleanup.
- Check interval: 60 seconds (hardcoded, not configurable).
- `acquire()` called from within `_get_whisperx_model()` (inside `_whisperx_model_lock`, atomic with model check).
- `release()` called by callers after model usage (try/finally in `/align`, after warmup).

### States added to `warmup_state`

- `"whisperx"`: `"evicted"` — new value, set by `IdleModelManager.unload()`.
- On re-load, `_mark_lazy_loaded("whisperx")` transitions back to `"ready"`.
- `"evicted"` is sanitized to `"evicted"` (safe for /health — no internal detail leaked).

### Config

| Env var | Default | Description |
|---------|---------|-------------|
| `AUTOMATIC_UNLOAD` | `true` | Master switch. Set to `false` to disable unloading entirely. |
| `MODEL_IDLE_TTL` | `300` | Seconds of inactivity before unload. `0` disables unloading. |

## Files

| File | Change |
|------|--------|
| `server.py` | Add `IdleModelManager` class, daemon thread in `_startup_event`, `acquire()`/`release()` calls in `_get_whisperx_model`, `"evicted"` in warmup_state |
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
5. **unload sets warmup_state["whisperx"] = "evicted"** — state transition.
6. **reload after unload** — `_get_whisperx_model()` reinitializes when `_whisperx_model is None`.
7. **health reflects "evicted"** — `"whisperx": "evicted"` in `/health` response.
8. **Default TTL is 300s** — no env var set.

## Tasks

### 1. `IdleModelManager` class — `server.py`

- Add `import threading` (already present), `time` (already present).
- Read `MODEL_IDLE_TTL` env var at module level.
- Define `IdleModelManager` class with `__init__`, `touch`, `unload`, `_check_loop`.
- Thread safety: use `_lock` around `_last_used` reads/writes.

### 2. Add "evicted" state support

- Add `"evicted"` to the set of possible `warmup_state` values.
- Ensure `_sanitize` in `/health` passes `"evicted"` through (it starts with no prefix, so it passes).
- Ensure `_mark_lazy_loaded` transitions "evicted" → "ready" correctly (it will: current check is `if current in ("ready", "downloading"): return`, and `"evicted"` is neither, so it will call `_set_warmup_state("whisperx", "ready")`).

### 3. Integrate into startup

- `IdleModelManager` already created at module level. Start its `_check_loop` daemon thread in `_startup_event`.

### 4. Integrate acquire into `_get_whisperx_model`

- At the end of `_get_whisperx_model()`, call manager's `acquire()` inside the model lock.



### 6. Integration test

- Write tests in `tests/test_idle_unload.py` using AST extraction.

### 7. Docker config

- Add `AUTOMATIC_UNLOAD=true`, `MODEL_IDLE_TTL=300` to `docker-compose.yml` environment section.
- Add `MODEL_IDLE_TTL=300` to `Dockerfile` ENV block.

## Implementation approach

TDD (red-green-refactor):

1. Write `tests/test_idle_unload.py` with all test scenarios
2. Implement `IdleModelManager` in `server.py`
3. Wire into `_get_whisperx_model`, `_startup_event`, `/align`, `/health`
4. Update Docker files
5. Run tests: `python -m pytest tests/test_idle_unload.py -v`
