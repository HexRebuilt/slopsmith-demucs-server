# Tasks

## Pre-existing (repo rename)
- [ ] Update `README.md`: Replace all occurrences of 'byrongamatos' with 'hexrebuilt' in Docker pull and git build URLs.
- [ ] Update `Dockerfile`: Replace 'byrongamatos' with 'hexrebuilt' in the 'org.opencontainers.image.source' label.
- [ ] Verify consistency across `README.md`, `Dockerfile`, and `docker-compose.yml`.

## Feature: Idle Model Unloading (unload WhisperX ASR after 5min inactivity)

### Task 1 — Write tests first (TDD red phase)
- [x] Create `tests/test_idle_unload.py` with AST extraction pattern

### Task 2 — Implement IdleModelManager class
- [x] Add `IdleModelManager` class in `server.py` with touch/unload/check_loop
- [x] Read `MODEL_IDLE_TTL` env var with safe int() parsing

### Task 3 — Wire into server lifecycle
- [x] Start daemon thread in `_startup_event` (conditional on TTL>0)
- [x] Call `manager.touch()` in `_get_whisperx_model()`
- [x] TOCTOU-safe unload with re-check under lock
- [x] Thread-safe _whisperx_model access with _whisperx_model_lock
- [x] _mark_lazy_loaded("whisperx") on model reload after unload

### Task 4 — Warmup state + health
- [x] "evicted" state transitions correctly in warmup_state
- [x] /health reflects "evicted" safely (no prefix leak)

### Task 5 — Docker config
- [x] Add `MODEL_IDLE_TTL=300` to `docker-compose.yml`
- [x] Add `MODEL_IDLE_TTL=300` to `Dockerfile`

### Task 6 — Green phase: tests pass
- [x] All 31 tests pass (13 new + 18 existing)
- [x] Full test suite green

## Post-merge (conditional on merge to fork/main)
- [ ] Change image ref in `docker-compose.yml` from `ghcr.io/byrongamatos/...` to `ghcr.io/hexrebuilt/...`
- [ ] Update Dockerfile LABEL
- [ ] Update README.md references
