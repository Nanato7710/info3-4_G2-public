from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

OUTPUTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = OUTPUTS_ROOT.parent
if str(OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_ROOT))

from artifact_common import (
    GENRE_NAMES,
    git_value,
    sha256_file,
    split_csv_path,
    utc_now,
    write_json,
)

MODELS = ("baseline", "final-tri-model")
SEEDS = (42, 43, 44)


def file_record(path: Path) -> dict:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    manifest_path = OUTPUTS_ROOT / "freeze-manifest.json"
    supersedes = None
    if manifest_path.is_file():
        supersedes = {
            "sha256": sha256_file(manifest_path),
            "reason": "frozen evaluation code changed; previous manifest is invalid",
        }
    required = [
        OUTPUTS_ROOT / "evaluation" / "results" / "validation-per-seed.csv",
        OUTPUTS_ROOT / "evaluation" / "results" / "validation-summary.json",
    ]
    frozen_inputs: list[Path] = []
    for model_id in MODELS:
        model_dir = OUTPUTS_ROOT / model_id
        frozen_inputs.extend(
            [
                model_dir / "config.yaml",
                *sorted(model_dir.glob("*.py")),
            ]
        )
        for seed in SEEDS:
            run_dir = model_dir / "runs" / f"seed_{seed}"
            frozen_inputs.extend(
                [
                    run_dir / "best_model.pth",
                    run_dir / "metrics.csv",
                    run_dir / "run-metadata.json",
                    run_dir / "resolved-config.yaml",
                    run_dir / "validation_predictions.npz",
                    run_dir / "validation_metrics.json",
                ]
            )
    frozen_inputs.extend(
        [
            OUTPUTS_ROOT / "artifact_common.py",
            OUTPUTS_ROOT / "run_train.sbatch",
            *sorted((OUTPUTS_ROOT / "evaluation").glob("*.py")),
            split_csv_path("training"),
            split_csv_path("validation"),
            split_csv_path("test"),
            *required,
        ]
    )
    missing = [path for path in frozen_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"freeze inputs are missing: {missing}")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(status),
        "dirty_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "dirty_status": status.splitlines(),
        "supersedes": supersedes,
        "threshold": 0.5,
        "genre_names": GENRE_NAMES,
        "checkpoint_selection": {
            "split": "validation",
            "metric": "mAP",
            "early_stopping_min_delta": 0.001,
            "test_used_for_selection": False,
        },
        "frozen_inputs": [
            file_record(path)
            for path in sorted(set(frozen_inputs), key=lambda item: item.as_posix())
        ],
    }
    write_json(manifest_path, manifest)
    print(
        f"created outputs/freeze-manifest.json with "
        f"{len(manifest['frozen_inputs'])} frozen inputs"
    )


if __name__ == "__main__":
    main()
