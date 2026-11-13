# Flaw Log

| Date | Suspicion | How I tested it | Result | Fix | Status |
| --- | --- | --- | --- | --- | --- |
| 2026-11-12 | CI run #1 failed after 9 seconds | Opened the failed run; `requirements.txt` still had the placeholder `pytest==<version installed in your venv>`, which pip cannot parse | Confirmed: dependency install crashed before any test ran | Pinned the real pytest version, CI #2 went green (commit b08f8f4) | Fixed |
| 2026-11-12 | Running a legacy script from the new setup might break | Ran `experiments/log_legacy_v1.py`; first attempt failed with "'experiments' is not a package" and also re-ran all old v1 experiments on import | Confirmed: `legacy/experiments.py` clashed with the `experiments/` package, and it has no `if __name__ == "__main__":` guard so it runs on import | Put `legacy/` last on `sys.path` (`append`, not `insert(0)`); left the frozen legacy files unchanged. Run reproduces v1 accuracy 0.9583 | Fixed |