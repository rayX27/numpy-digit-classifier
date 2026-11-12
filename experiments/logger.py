import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def log_result(experiment, metrics, config, seconds=None,
               path=ROOT / "results" / "log.jsonl"):
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
        "metrics": metrics,
        "config": config.to_dict(),
        "wall_clock_s": seconds,
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": _git("status", "--porcelain") != "",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")