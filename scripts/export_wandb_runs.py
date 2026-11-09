"""Read-only export of every run's config, summary and unsampled metric history.

Uses the existing W&B login. Never accepts or writes an API key. Downloaded
histories are private local inputs; reviewed aggregates belong under docs/analysis.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re


def redact(value):
    if isinstance(value, dict):
        return {
            key: "[redacted]" if re.search(
                r"(^|_)(api_key|access_token|password|secret|authorization)($|_)",
                key, re.I,
            ) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="xlz1014361885-tokyo/ppo-snake")
    parser.add_argument("--output", type=Path, default=Path("analysis/wandb/raw"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = Path("target/wandb-analysis").resolve()
    os.environ.setdefault("WANDB_CONFIG_DIR", str(cache / "config"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache / "cache"))
    import wandb

    api = wandb.Api(timeout=45)
    runs = list(api.runs(args.project, per_page=100, order="+created_at"))
    inventory = {
        "project": args.project,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "wandb.Api.runs + Run.scan_history(keys=None, use_cache=False); no UI sampling",
        "run_count": len(runs), "runs": [],
    }
    print(f"Found {len(runs)} runs", flush=True)
    failures = []
    for i, run in enumerate(runs, 1):
        folder = args.output / run.id
        folder.mkdir(exist_ok=True)
        metadata_path = folder / "metadata.json"
        history_path = folder / "history.jsonl.gz"
        metadata = {
            "id": run.id, "name": run.name, "group": run.group,
            "state": run.state, "url": run.url,
            "created_at": run.created_at, "updated_at": run._attrs.get("updatedAt"),
            "config": redact(run.config), "summary": redact(run.summary._json_dict),
        }
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            if (not args.refresh and cached.get("export_complete") and history_path.exists()
                    and cached.get("updated_at") == metadata["updated_at"]
                    and cached.get("summary") == metadata["summary"]):
                metadata = cached
                print(f"[{i}/{len(runs)}] cached {run.id} {run.name}", flush=True)
            else:
                partial = folder / "history.partial.jsonl.gz"
                count, keys, max_samples, max_step = 0, set(), None, None
                with gzip.open(partial, "wt", encoding="utf-8") as stream:
                    for row in run.scan_history(page_size=2000, use_cache=False):
                        stream.write(json.dumps(redact(row), ensure_ascii=False, default=str) + "\n")
                        count += 1
                        keys.update(row)
                        if isinstance(row.get("total_samples"), (int, float)):
                            max_samples = max(max_samples or 0, row["total_samples"])
                        if isinstance(row.get("_step"), (int, float)):
                            max_step = max(max_step or 0, row["_step"])
                partial.replace(history_path)
                metadata.update({
                    "export_complete": True, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                    "history_rows": count, "history_keys": sorted(keys),
                    "max_history_total_samples": max_samples, "max_history_step": max_step,
                    "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
                })
                write_json(metadata_path, metadata)
                print(f"[{i}/{len(runs)}] {run.id} {run.name}: {count} rows, {max_samples} samples", flush=True)
            inventory["runs"].append({key: metadata.get(key) for key in [
                "id", "name", "group", "state", "url", "exported_at_utc", "history_rows",
                "history_sha256", "max_history_total_samples", "max_history_step",
            ]})
        except Exception as error:
            failures.append(run.id)
            inventory["runs"].append({"id": run.id, "name": run.name, "error": str(error)[:500]})
            print(f"ERROR {run.id}: {type(error).__name__}: {str(error)[:180]}", flush=True)
        write_json(args.output / "inventory.json", inventory)
    inventory["export_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    inventory["failed_runs"] = failures
    write_json(args.output / "inventory.json", inventory)
    if failures:
        raise SystemExit(f"Incomplete export: {len(failures)} runs; rerun to retry")


if __name__ == "__main__":
    main()
