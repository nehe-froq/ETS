from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Optional

import yaml


def load_config(path: str | None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch CSV and run incremental ingest on changes")
    parser.add_argument("--csv", required=False, default="./embeddings/data.csv", help="CSV path")
    parser.add_argument("--out", required=False, default="./embeddings/store", help="Index directory")
    parser.add_argument("--config", required=False, default=None, help="YAML config path")
    parser.add_argument("--debounce-ms", type=int, default=None, help="Debounce window in ms")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path = cfg.get("csv", {}).get("path", args.csv)
    debounce_ms = args.debounce_ms if args.debounce_ms is not None else int(cfg.get("watch", {}).get("debounce_ms", 1500))

    print(f"Watching {csv_path} with debounce {debounce_ms}ms. Press Ctrl+C to stop.")

    last_mtime: Optional[float] = None
    pending = False
    last_event_ts = 0.0
    try:
        while True:
            try:
                stat = os.stat(csv_path)
            except FileNotFoundError:
                time.sleep(0.5)
                continue
            if last_mtime is None or stat.st_mtime != last_mtime:
                last_mtime = stat.st_mtime
                pending = True
                last_event_ts = time.time()
            if pending and (time.time() - last_event_ts) * 1000.0 >= debounce_ms:
                print("Change detected. Running incremental ingest...")
                cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "ingest_csv.py"), "--csv", csv_path, "--out", args.out, "--incremental"]
                res = subprocess.run(cmd)
                if res.returncode != 0:
                    print(f"Ingest failed with code {res.returncode}")
                else:
                    print("Ingest done.")
                pending = False
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


