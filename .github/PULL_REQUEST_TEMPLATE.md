## Summary
-

## Why
-

## Validation
- [ ] Focused validation: `<command>` -> `<result>`
- [ ] Default validation:
  - [ ] `pytest tests/test_lcm_core.py tests/test_lcm_engine.py tests/test_packaging_install.py -q`
  - [ ] `pytest -q`
  - [ ] `bash -lc 'ulimit -n 1024 && pytest -q'`
  - [ ] `python -m compileall -q .`
  - [ ] `python -m py_compile scripts/import_lossless_claw.py`
  - [ ] `bash -n scripts/install.sh scripts/update.sh`
  - [ ] `git diff --check`
- [ ] Workflow validation, if workflows changed: `actionlint`

## Notes
-

Refs #
