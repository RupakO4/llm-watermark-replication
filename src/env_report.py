#!/usr/bin/env python3
"""Record the software/hardware environment of a run (-> results/environment.json)."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import REPO_ROOT, UPSTREAM_DIR, UPSTREAM_COMMIT  # noqa: E402


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main():
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor()}
    if sys.platform == "darwin":
        info["cpu"] = sh("sysctl -n machdep.cpu.brand_string")
        mem = sh("sysctl -n hw.memsize")
        info["ram_gb"] = round(int(mem) / 2**30, 1) if mem.isdigit() else None
        info["macos"] = sh("sw_vers -productVersion")
    else:
        info["cpu"] = sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2")
        mem = sh("grep MemTotal /proc/meminfo | awk '{print $2}'")
        info["ram_gb"] = round(int(mem) / 2**20, 1) if mem.isdigit() else None
    info["cpu_count"] = os.cpu_count()
    for mod in ("torch", "transformers", "datasets", "tokenizers", "scipy", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        import torch
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        info["cuda_available"] = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        pass
    info["upstream_dir"] = str(UPSTREAM_DIR)
    info["upstream_commit_expected"] = UPSTREAM_COMMIT
    info["upstream_commit_found"] = sh(f"git -C '{UPSTREAM_DIR}' rev-parse HEAD")
    info["repo_commit"] = sh(f"git -C '{REPO_ROOT}' rev-parse --short HEAD")
    out = REPO_ROOT / "results" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    if info["upstream_commit_found"] and info["upstream_commit_found"] != UPSTREAM_COMMIT:
        print("WARNING: upstream lm-watermarking is not at the pinned commit")


if __name__ == "__main__":
    main()
