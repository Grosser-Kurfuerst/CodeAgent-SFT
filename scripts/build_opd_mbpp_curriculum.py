#!/usr/bin/env python3
"""Build small MBPP Teach-to-Transfer manifests for the 16GB OPD workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROUND_TASK_IDS = {
    0: (
        # Regex: teach one search pattern, then transfer to related validation/cleanup tasks.
        "mbpp_607", "mbpp_631", "mbpp_669", "mbpp_676",
        # Prime-number reasoning.
        "mbpp_605", "mbpp_681", "mbpp_685", "mbpp_849",
        # Consecutive-item scans and grouping.
        "mbpp_730", "mbpp_778", "mbpp_819", "mbpp_766",
        # Pairwise tuple operations that commonly reuse zip-based traversal.
        "mbpp_616", "mbpp_695", "mbpp_809", "mbpp_888",
    ),
    1: (
        # Held-out regex transfers.
        "mbpp_693", "mbpp_725", "mbpp_727", "mbpp_737",
        # Tuple aggregation transfers.
        "mbpp_611", "mbpp_613", "mbpp_614", "mbpp_615",
        # Duplicate-removal transfers.
        "mbpp_674", "mbpp_712", "mbpp_932", "mbpp_929",
        # Additional pairwise/adjacent tuple operations.
        "mbpp_630", "mbpp_726", "mbpp_740", "mbpp_951",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/sft_raw/mbpp_train/tasks/mbpp_tasks.jsonl"),
        help="MBPP train task pool JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sft_raw/mbpp_train/tasks"),
        help="Directory for opd_round0_tasks.json and opd_round1_tasks.json.",
    )
    args = parser.parse_args()

    tasks = _load_tasks(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for round_index, task_ids in ROUND_TASK_IDS.items():
        payload = _round_manifest(round_index, task_ids, tasks)
        target = args.output_dir / f"opd_round{round_index}_tasks.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(task_ids)} tasks to {target}")
    return 0


def _load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
                raise ValueError(f"invalid MBPP task at line {line_number}")
            task_id = payload["id"]
            if task_id in tasks:
                raise ValueError(f"duplicate MBPP task ID: {task_id}")
            tasks[task_id] = payload
    return tasks


def _round_manifest(
    round_index: int,
    task_ids: tuple[str, ...],
    tasks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing = [task_id for task_id in task_ids if task_id not in tasks]
    if missing:
        raise ValueError(f"MBPP curriculum tasks are missing from the train pool: {missing}")
    return {
        "memory_mode": "shared_stream",
        "stream_id": f"mbpp-opd-round-{round_index}",
        "task_group_fallback": "source",
        "tasks": [
            {
                "id": task_id,
                "source": str(tasks[task_id].get("source") or "MBPP"),
                "repo": f"../repos/mbpp/{task_id}",
                "task": str(tasks[task_id]["task"]),
                "test_command": "python -m pytest -q",
                "success_hint": str(
                    tasks[task_id].get("success_hint")
                    or "All generated pytest tests pass."
                ),
                "task_group": "mbpp",
            }
            for task_id in task_ids
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
