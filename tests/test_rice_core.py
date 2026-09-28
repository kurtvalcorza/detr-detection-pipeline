"""Hand-computed scientific contracts; synthetic fixtures are not dataset evidence."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("rice_core", Path(__file__).parents[1] / "tools/rice_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def fixture_manifest():
    return {
        "classes": list(core.SPECIES),
        "records": [
            {
                "image_id": "one",
                "relative_path": "images/one.png",
                "sha256": "a" * 64,
                "bytes": 20,
                "width": 100,
                "height": 100,
                "capture_group_id": "g1",
                "split": "train",
                "source_record": "synthetic-test-only",
                "licence": "test-only",
                "attribution": "test",
                "annotation_status": "human_verified_complete",
                "objects": [
                    {
                        "object_id": "o1",
                        "bbox_xyxy": [0, 0, 20, 20],
                        "species": core.SPECIES[0],
                        "original_label": "rbb",
                        "ignore": False,
                    }
                ],
            }
        ],
        "crops": [
            {
                "crop_id": "c1",
                "parent_image_id": "one",
                "parent_object_id": "o1",
                "bbox_xyxy": [0, 0, 20, 20],
                "preprocessing": "reference-box-v1",
            }
        ],
    }


def test_counts_include_zeros_and_fractional_baselines():
    result = core.count_metrics([[0, 2], [2, 0]], [[0, 1], [3, 0]])
    assert result["mae"] == 0.5
    assert result["bias"] == 0
    assert result["exact_count_rate"] == 0.5
    assert result["total_target_mae"] == 1
    assert result["per_species_bias"] == [0.5, -0.5]
    assert core.count_metrics([[0, 0]], [[0.5, 0.5]])["mae"] == 0.5


@pytest.mark.parametrize(
    "truth,prediction",
    [
        ([], []),
        ([[0, 0]], [[-1, 0]]),
        ([[0.5, 0]], [[0, 0]]),
        ([[0, 0]], [[float("nan"), 0]]),
        ([[0, 0]], [[0]]),
    ],
)
def test_invalid_counts_refused(truth, prediction):
    with pytest.raises(ValueError):
        core.count_metrics(truth, prediction)


def test_matching_cardinality_precedes_largest_single_iou():
    # R0 matches P0 perfectly but must take P1 to allow R1's only possible match.
    result = core.match_boxes([[0, 0, 10, 10], [3, 0, 13, 10]], [[0, 0, 10, 10], [-3, 0, 7, 10]])
    assert [(r, p) for r, p, _ in result["matches"]] == [(0, 1), (1, 0)]
    assert result["misses"] == result["spurious"] == []


def test_matching_secondary_iou_objective_and_empty_cases():
    boxes = [[0, 0, 10, 10], [1, 0, 11, 10]]
    result = core.match_boxes(boxes, boxes[::-1])
    assert [(r, p) for r, p, _ in result["matches"]] == [(0, 1), (1, 0)]
    assert core.match_boxes([], boxes)["spurious"] == [0, 1]
    assert core.match_boxes(boxes, [])["misses"] == [0, 1]
    assert core.match_boxes([], [])["matches"] == []


def test_duplicate_predictions_and_wrong_species_decompose():
    ref = [
        {"bbox_xyxy": [0, 0, 10, 10], "species": core.SPECIES[0]},
        {"bbox_xyxy": [30, 0, 40, 10], "species": core.SPECIES[1]},
    ]
    pred = [{"bbox_xyxy": [0, 0, 10, 10], "species": core.SPECIES[1]}] * 2
    result = core.error_decomposition(ref, pred)
    assert (result["missed"], result["spurious_count"], result["wrong_species"]) == (1, 1, 1)
    ref[0]["ignore"] = True
    with pytest.raises(ValueError, match="Ignore"):
        core.error_decomposition(ref, pred)


def test_threshold_requires_validation_grid_and_ties_go_high():
    rows = [{"threshold": i / 20, "mae": 1.0} for i in range(1, 20)]
    assert core.select_count_threshold(rows)["threshold"] == 0.95
    rows[5]["mae"] = 0.5
    assert core.select_count_threshold(rows)["threshold"] == 0.3
    with pytest.raises(ValueError):
        core.select_count_threshold(rows, split="test")
    with pytest.raises(ValueError):
        core.select_count_threshold(rows[:-1])


def test_paired_bootstrap_retains_groups_and_pairing():
    truth = np.zeros((4, 2))
    a = np.array([[1, 1], [1, 1], [1, 1], [3, 3]])
    b = a + 2
    result = core.paired_group_bootstrap(truth, a, b, ["a", "a", "a", "b"], n_boot=100)
    assert result["difference_ci"] == [-2, -2]
    assert result["mae_a_ci"] == [1, 3]
    assert result["difference"] == -2
    assert result == core.paired_group_bootstrap(truth, a, b, ["a", "a", "a", "b"], n_boot=100)


def test_valid_schema_does_not_claim_qualification():
    result = core.validate_manifest(fixture_manifest())
    assert result["objects"] == 1
    assert result["dataset_qualified"] is False
    with pytest.raises(ValueError, match="Insufficient"):
        core.validate_manifest(fixture_manifest(), require_capacity=True)


def test_exploratory_scope_must_be_explicit_and_cannot_certify_capture_capacity():
    manifest = fixture_manifest()
    manifest["records"][0]["annotation_status"] = "published_unverified"
    with pytest.raises(ValueError, match="human-verified"):
        core.validate_manifest(manifest)
    manifest["scope"] = "exploratory_published_annotations"
    result = core.validate_manifest(manifest)
    assert len(result["limitations"]) == 2
    assert result["dataset_qualified"] is False
    with pytest.raises(ValueError, match="cannot certify"):
        core.validate_manifest(manifest, require_capacity=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m["records"][0].update(annotation_status="detector_generated"),
        lambda m: m["records"][0]["objects"][0].update(bbox_xyxy=[0, 0, 101, 20]),
        lambda m: m["records"][0]["objects"][0].update(species="unknown"),
        lambda m: m["records"][0]["objects"][0].pop("ignore"),
        lambda m: m["records"][0]["objects"][0].update(ignore=True),
        lambda m: m["crops"][0].update(parent_object_id="missing"),
        lambda m: m["crops"][0].update(split="test"),
        lambda m: m["records"][0].update(relative_path="../image.png"),
    ],
)
def test_invalid_schema_refused(mutation):
    manifest = fixture_manifest()
    mutation(manifest)
    with pytest.raises(ValueError):
        core.validate_manifest(manifest)


@pytest.mark.parametrize(
    "link",
    [
        "capture_group_id",
        "sha256",
        "decoded_sha256",
        "duplicate_cluster_id",
        "capture_session_id",
        "specimen_id",
        "tray_id",
        "burst_id",
    ],
)
def test_every_lineage_link_prevents_leakage(link):
    manifest = fixture_manifest()
    first = manifest["records"][0]
    first[link] = "a" * 64 if "sha256" in link else "shared"
    second = copy.deepcopy(first)
    second.update(
        image_id="two",
        relative_path="images/two.png",
        split="test",
        objects=[],
        capture_group_id="g2",
        sha256="b" * 64,
    )
    second[link] = first[link]
    manifest["records"].append(second)
    with pytest.raises(ValueError, match="Cross-split"):
        core.validate_manifest(manifest)


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "x\\y", "x/./y", "x//y", ""])
def test_archive_paths_refused(tmp_path, name):
    with pytest.raises(ValueError):
        core.safe_path(tmp_path, name)


def test_digest_is_order_independent_and_nonfinite_refused():
    assert core.digest_json({"a": 1, "b": 2}) == core.digest_json({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        core.digest_json({"bad": float("nan")})
