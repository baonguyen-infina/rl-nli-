"""Synthetic fixtures for the sealed held-out evaluation manifest and receipts."""
import copy
import hashlib
import json

import pytest
import yaml

from eval.final_protocol import (
    claim_test_run,
    complete_test_run,
    sha256_path,
    validate_protocol,
)


PROTOCOL_SHA = "a" * 64
OTHER_PROTOCOL_SHA = "b" * 64


def _fixture(root):
    run_dir = root / "results" / "N4R9_Qwen3.5-2B_s0"
    run_dir.mkdir(parents=True)
    cfg = {"run_id": "N4R9", "seeds": [0], "model": {"name": "Qwen/Qwen3.5-2B"},
           "recipe": "native", "format": "cot"}
    config = run_dir / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    adapter = run_dir / "best_adapter" / "weights.bin"
    adapter.parent.mkdir()
    adapter.write_bytes(b"synthetic student adapter")
    manifest = {
        "schema_version": 1,
        "locked": True,
        "selection_split": "dev",
        "report_split": "test",
        "runs": [{
            "run_id": "N4R9", "seed": 0, "model": "Qwen/Qwen3.5-2B",
            "recipe": "native", "format": "cot",
            "source_config": {"path": str(config.relative_to(root)),
                              "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
            "adapter": {"path": str(adapter.relative_to(root)),
                        "sha256": hashlib.sha256(b"synthetic student adapter").hexdigest()},
        }],
    }
    return manifest, run_dir


def _outputs(output_dir, *, n=2):
    output_dir.mkdir(parents=True, exist_ok=True)
    # These are synthetic predictions, never real held-out examples.
    predictions = [{"id": "synthetic-a", "gold": "neutral", "pred": "neutral"},
                   {"id": "synthetic-b", "gold": "entailment", "pred": "neutral"}]
    (output_dir / "predictions_test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    (output_dir / "metrics_test.json").write_text(
        json.dumps({"n": n, "macro_f1": 0.5, "accuracy": 0.5, "parse_rate": 1.0}),
        encoding="utf-8")


def test_file_digest_is_standard_sha256(tmp_path):
    path = tmp_path / "synthetic.bin"
    path.write_bytes(b"abc")
    assert sha256_path(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_valid_protocol_resolves_artifacts_without_mutating_manifest(tmp_path):
    manifest, run_dir = _fixture(tmp_path)
    original = copy.deepcopy(manifest)

    entries = validate_protocol(manifest, root=tmp_path)

    assert len(entries) == 1
    assert entries[0]["source_config_path"] == run_dir / "config.yaml"
    assert entries[0]["evaluation_config_path"] == run_dir / "config.yaml"
    assert entries[0]["adapter_path"] == run_dir / "best_adapter" / "weights.bin"
    assert entries[0]["evaluation_config_data"]["format"] == "cot"
    assert manifest == original


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("locked", False),
    ("locked", "true"), ("selection_split", "dev_mini"), ("report_split", "dev"),
    ("runs", []), ("runs", {}),
])
def test_incomplete_or_unlocked_protocol_is_rejected(tmp_path, field, value):
    manifest, _ = _fixture(tmp_path)
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


def test_duplicate_run_and_seed_are_rejected(tmp_path):
    manifest, _ = _fixture(tmp_path)
    manifest["runs"].append(copy.deepcopy(manifest["runs"][0]))
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


@pytest.mark.parametrize("field,value", [
    ("run_id", "../N4R9"), ("seed", True), ("seed", -1),
    ("model", "other/model"), ("format", "label"), ("recipe", "zs"),
])
def test_run_identity_and_source_configuration_must_agree(tmp_path, field, value):
    manifest, _ = _fixture(tmp_path)
    manifest["runs"][0][field] = value
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside", "results/../../outside",
                                  "configs/base.yaml", "results\\outside"])
def test_artifact_paths_cannot_escape_results(tmp_path, path):
    manifest, _ = _fixture(tmp_path)
    manifest["runs"][0]["adapter"]["path"] = path
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


@pytest.mark.parametrize("artifact", ["source_config", "adapter"])
def test_changed_artifact_content_invalidates_protocol(tmp_path, artifact):
    manifest, _ = _fixture(tmp_path)
    path = tmp_path / manifest["runs"][0][artifact]["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


def test_missing_artifact_is_rejected(tmp_path):
    manifest, _ = _fixture(tmp_path)
    (tmp_path / manifest["runs"][0]["adapter"]["path"]).unlink()
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


def test_symlinked_adapter_cannot_escape_results(tmp_path):
    manifest, run_dir = _fixture(tmp_path)
    external = tmp_path / "outside.bin"
    external.write_bytes(b"synthetic student adapter")
    link = run_dir / "best_adapter" / "outside-link.bin"
    link.symlink_to(external)
    manifest["runs"][0]["adapter"]["path"] = str(link.relative_to(tmp_path))
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


def test_adapter_from_another_seed_is_rejected_even_with_correct_digest(tmp_path):
    manifest, run_dir = _fixture(tmp_path)
    other = tmp_path / "results" / "D3R9_Qwen3.5-2B_s1" / "weights.bin"
    other.parent.mkdir()
    other.write_bytes((run_dir / "best_adapter" / "weights.bin").read_bytes())
    manifest["runs"][0]["adapter"]["path"] = str(other.relative_to(tmp_path))
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


def test_same_seed_warm_start_adapter_is_allowed(tmp_path):
    manifest, run_dir = _fixture(tmp_path)
    warm = tmp_path / "results" / "D3R9_Qwen3.5-2B_s0" / "weights.bin"
    warm.parent.mkdir()
    warm.write_bytes((run_dir / "best_adapter" / "weights.bin").read_bytes())
    manifest["runs"][0]["adapter"]["path"] = str(warm.relative_to(tmp_path))
    assert validate_protocol(manifest, root=tmp_path)[0]["adapter_path"] == warm


def test_directory_digest_protects_adapter_configuration_and_weights(tmp_path):
    manifest, run_dir = _fixture(tmp_path)
    adapter = run_dir / "best_adapter"
    configuration = adapter / "adapter_config.json"
    configuration.write_text('{"r":32}', encoding="utf-8")
    manifest["runs"][0]["adapter"] = {
        "path": str(adapter.relative_to(tmp_path)), "sha256": sha256_path(adapter)}
    assert validate_protocol(manifest, root=tmp_path)[0]["adapter_path"] == adapter
    configuration.write_text('{"r":64}', encoding="utf-8")
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


def test_frozen_evaluation_view_can_change_format_and_recipe(tmp_path):
    manifest, run_dir = _fixture(tmp_path)
    effective = {"run_id": "N4R9", "seeds": [0], "model": {"name": "Qwen/Qwen3.5-2B"},
                 "recipe": "zs", "format": "label", "eval": {"backend": "transformers"}}
    path = run_dir / "test_eval_config.yaml"
    path.write_text(yaml.safe_dump(effective), encoding="utf-8")
    manifest["runs"][0].update(recipe="zs", format="label", evaluation_config={
        "path": str(path.relative_to(tmp_path)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    selected = validate_protocol(manifest, root=tmp_path)[0]

    assert selected["evaluation_config_path"] == path
    assert selected["source_config_data"]["recipe"] == "native"
    assert selected["evaluation_config_data"]["recipe"] == "zs"
    assert selected["evaluation_config_data"]["format"] == "label"


def test_evaluation_view_must_remain_in_named_run_directory(tmp_path):
    manifest, run_dir = _fixture(tmp_path)
    other = tmp_path / "results" / "other.yaml"
    other.write_bytes((run_dir / "config.yaml").read_bytes())
    manifest["runs"][0]["evaluation_config"] = {
        "path": str(other.relative_to(tmp_path)), "sha256": hashlib.sha256(other.read_bytes()).hexdigest()}
    with pytest.raises(ValueError):
        validate_protocol(manifest, root=tmp_path)


def test_first_claim_is_exclusive_and_in_progress_claim_cannot_restart(tmp_path):
    claimed = claim_test_run(tmp_path, PROTOCOL_SHA)
    assert claimed["reused"] is False
    assert claimed["receipt"]["status"] == "in_progress"
    with pytest.raises((ValueError, FileExistsError)):
        claim_test_run(tmp_path, PROTOCOL_SHA)


def test_pending_atomic_claim_without_receipt_fails_closed(tmp_path):
    (tmp_path / ".test_claim").mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        claim_test_run(tmp_path, PROTOCOL_SHA)


def test_claim_refuses_unattributed_existing_test_outputs(tmp_path):
    _outputs(tmp_path)
    with pytest.raises((ValueError, FileExistsError)):
        claim_test_run(tmp_path, PROTOCOL_SHA)


def test_completion_records_actual_count_and_reuses_verified_outputs(tmp_path):
    claim_test_run(tmp_path, PROTOCOL_SHA)
    _outputs(tmp_path)
    (tmp_path / "cost_test.json").write_text('{"wall_seconds":1.0}', encoding="utf-8")

    receipt = complete_test_run(tmp_path, PROTOCOL_SHA)
    reused = claim_test_run(tmp_path, PROTOCOL_SHA)

    assert receipt["status"] == "completed"
    assert receipt["n"] == 2
    assert receipt["manifest_sha256"] == PROTOCOL_SHA
    assert set(receipt["artifacts"]) == {"predictions_test.jsonl", "metrics_test.json", "cost_test.json"}
    assert reused["reused"] is True
    assert reused["receipt"] == receipt


def test_different_protocol_cannot_reuse_or_complete_a_claim(tmp_path):
    claim_test_run(tmp_path, PROTOCOL_SHA)
    _outputs(tmp_path)
    with pytest.raises(ValueError):
        complete_test_run(tmp_path, OTHER_PROTOCOL_SHA)
    complete_test_run(tmp_path, PROTOCOL_SHA)
    with pytest.raises(ValueError):
        claim_test_run(tmp_path, OTHER_PROTOCOL_SHA)


@pytest.mark.parametrize("artifact", ["predictions_test.jsonl", "metrics_test.json"])
def test_changed_completed_result_cannot_be_silently_reused(tmp_path, artifact):
    claim_test_run(tmp_path, PROTOCOL_SHA)
    _outputs(tmp_path)
    complete_test_run(tmp_path, PROTOCOL_SHA)
    path = tmp_path / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        claim_test_run(tmp_path, PROTOCOL_SHA)


def test_completion_rejects_missing_outputs(tmp_path):
    claim_test_run(tmp_path, PROTOCOL_SHA)
    with pytest.raises(ValueError):
        complete_test_run(tmp_path, PROTOCOL_SHA)


def test_completion_never_invents_n_or_accepts_truncated_predictions(tmp_path):
    claim_test_run(tmp_path, PROTOCOL_SHA)
    _outputs(tmp_path, n=1000)
    with pytest.raises(ValueError):
        complete_test_run(tmp_path, PROTOCOL_SHA)


def test_completion_rejects_duplicate_prediction_ids(tmp_path):
    claim_test_run(tmp_path, PROTOCOL_SHA)
    _outputs(tmp_path)
    path = tmp_path / "predictions_test.jsonl"
    row = '{"id":"duplicate","gold":"neutral","pred":"neutral"}\n'
    path.write_text(row + row, encoding="utf-8")
    with pytest.raises(ValueError):
        complete_test_run(tmp_path, PROTOCOL_SHA)


@pytest.mark.parametrize("digest", ["", "not-a-hash", "g" * 64])
def test_claim_rejects_malformed_protocol_digest(tmp_path, digest):
    with pytest.raises(ValueError):
        claim_test_run(tmp_path, digest)
