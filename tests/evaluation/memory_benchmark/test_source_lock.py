from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import subprocess

import pytest

from my_agent.evaluation.memory_benchmark.source_lock import (
    SOURCE_LOCK_SCHEMA_VERSION,
    SourceLockError,
    load_source_lock,
    validate_source_lock,
)


FIXTURE_PATH = Path("tests/data/memory_benchmark/source_lock.json")


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_valid_source_lock_loads() -> None:
    payload = load_source_lock(FIXTURE_PATH)

    assert payload["schema_version"] == SOURCE_LOCK_SCHEMA_VERSION
    assert set(payload["sources"]) == {"lifelong_agent_bench", "intercode"}


def test_valid_multi_file_task_data_is_accepted() -> None:
    payload = _fixture()
    source = payload["sources"]["intercode"]
    del source["task_data_path"]
    source.pop("task_data_sha256", None)
    source["task_data_files"] = [
        {
            "path": "data/nl2bash/nl2bash_fs_1.json",
            "sha256": f"sha256:{'1' * 64}",
        },
        {
            "path": "data/nl2bash/nl2bash_fs_2.json",
            "sha256": f"sha256:{'2' * 64}",
        },
    ]

    validate_source_lock(payload)


@pytest.mark.parametrize(
    ("task_data_files", "match"),
    [
        ([], "must not be empty"),
        (
            [
                {"path": "data/tasks.json", "sha256": f"sha256:{'1' * 64}"},
                {"path": "data/tasks.json", "sha256": f"sha256:{'2' * 64}"},
            ],
            "duplicate path",
        ),
        (
            [{"path": "data/tasks.json", "sha256": "sha256:invalid"}],
            "sha256",
        ),
    ],
)
def test_invalid_multi_file_task_data_is_rejected(
    task_data_files: list[dict[str, str]],
    match: str,
) -> None:
    payload = _fixture()
    source = payload["sources"]["intercode"]
    del source["task_data_path"]
    source.pop("task_data_sha256", None)
    source["task_data_files"] = task_data_files

    with pytest.raises(SourceLockError, match=match):
        validate_source_lock(payload)


def test_single_and_multi_file_task_data_cannot_be_combined() -> None:
    payload = _fixture()
    payload["sources"]["intercode"]["task_data_files"] = [
        {
            "path": "data/nl2bash/nl2bash_fs_2.json",
            "sha256": f"sha256:{'2' * 64}",
        }
    ]

    with pytest.raises(SourceLockError, match="exactly one"):
        validate_source_lock(payload)


@pytest.mark.parametrize("revision", ["", "main", "master", "latest", "abc123"])
def test_moving_or_incomplete_revision_is_rejected(revision: str) -> None:
    payload = _fixture()
    payload["sources"]["intercode"]["revision"] = revision

    with pytest.raises(SourceLockError, match="revision"):
        validate_source_lock(payload)


def test_missing_container_digest_is_rejected() -> None:
    payload = _fixture()
    del payload["sources"]["intercode"]["container_digest"]

    with pytest.raises(SourceLockError, match="container_digest"):
        validate_source_lock(payload)


@pytest.mark.parametrize("image", ["intercode-nl2bash:latest", "repo/image@latest"])
def test_latest_container_image_is_rejected(image: str) -> None:
    payload = _fixture()
    payload["sources"]["intercode"]["container_image"] = image

    with pytest.raises(SourceLockError, match="latest"):
        validate_source_lock(payload)


def test_empty_license_is_rejected() -> None:
    payload = _fixture()
    payload["sources"]["intercode"]["license"] = ""

    with pytest.raises(SourceLockError, match="license"):
        validate_source_lock(payload)


def test_checkout_head_mismatch_is_rejected(tmp_path: Path) -> None:
    checkout = tmp_path / "intercode"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Memory Benchmark Tests",
            "-c",
            "user.email=memory-benchmark@example.test",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    payload = deepcopy(_fixture())

    with pytest.raises(SourceLockError, match="HEAD mismatch"):
        validate_source_lock(payload, checkout_roots={"intercode": checkout})


def test_checkout_head_match_is_accepted(tmp_path: Path) -> None:
    checkout = tmp_path / "intercode"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Memory Benchmark Tests",
            "-c",
            "user.email=memory-benchmark@example.test",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = _fixture()
    payload["sources"]["intercode"]["revision"] = head

    validate_source_lock(payload, checkout_roots={"intercode": checkout})
