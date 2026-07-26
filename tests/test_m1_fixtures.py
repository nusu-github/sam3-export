"""Contract checks for the locked M1 comparison inputs and policy cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "m1_image_pcs" / "cases.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_m1_real_images_and_required_case_coverage_are_locked() -> None:
    fixture = _fixture()
    assert fixture["fixture_version"] == "m1-image-pcs-v1"
    for image in fixture["images"]:
        assert _sha256(WORKSPACE / image["workspace_path"]) == image["sha256"]
    coverage = {tag for case in fixture["cases"] for tag in case["coverage"]}
    assert {"single-instance", "multiple-instance", "no-match"} <= coverage


def test_m1_threshold_and_stable_topk_policy_cases() -> None:
    cases = {case["id"]: case for case in _fixture()["policy_cases"]}
    threshold = cases["threshold_boundary"]
    admitted = np.flatnonzero(
        np.asarray(threshold["scores"]) > threshold["threshold"]
    ).tolist()
    assert admitted == threshold["expected_indices"]

    topk = cases["topk_tie"]
    scores = np.asarray(topk["scores"])
    order = np.lexsort((np.arange(scores.size), -scores))[: topk["k"]].tolist()
    assert order == topk["expected_indices"]
