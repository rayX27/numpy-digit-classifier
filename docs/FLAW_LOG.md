# Flaw Log

| Date | Suspicion | How I tested it | Result | Fix | Status |
| --- | --- | --- | --- | --- | --- |
| 2026-11-12 | CI run #1 failed after 9 seconds | Opened the failed run; `requirements.txt` still had the placeholder `pytest==<version installed in your venv>`, which pip cannot parse | Confirmed: dependency install crashed before any test ran | Pinned the real pytest version, CI #2 went green (commit b08f8f4) | Fixed |