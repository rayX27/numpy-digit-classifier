import json
import subprocess
import sys
import time
import platform
import numpy as np
from pathlib import Path
import sys
import os

# Ensure we can import config from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def log_result(experiment_name: str, metrics: dict, filepath: str = "results/log.jsonl"):
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": experiment_name,
        "metrics": metrics,
        "config": {
            "seed": config.seed,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "hidden_sizes": config.hidden_sizes
        },
        "provenance": {
            "git_hash": get_git_hash(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "platform": platform.platform()
        }
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")