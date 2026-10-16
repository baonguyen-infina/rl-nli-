import pytest
from train.common import build_sft_dataset, build_grpo_dataset

ROWS = [{"id": "uit_Adver_1_1_11_01", "dataset": "vianli", "premise": "P1", "hypothesis": "H1",
         "label": "neutral", "topic": "01", "premise_id": "1", "intended_label": "entailment"}]


def test_sft_dataset_label():
    ds = build_sft_dataset(ROWS, "label")
    assert ds[0]["completion"] == [{"role": "assistant", "content": "<answer>neutral</answer>"}]
    assert ds[0]["prompt"][0]["role"] == "user" and "P1" in ds[0]["prompt"][0]["content"]
    assert set(ds[0]) == {"prompt", "completion"}


def test_no_uid_or_intended_label_leak():
    for ds in (build_sft_dataset(ROWS, "label"), build_grpo_dataset(ROWS, "label")):
        blob = str(ds)
        assert "uit_Adver" not in blob
        assert "intended" not in blob


def test_grpo_dataset_columns():
    ds = build_grpo_dataset(ROWS, "label")
    assert ds[0]["gold"] == "neutral" and ds[0]["premise"] == "P1" and ds[0]["hypothesis"] == "H1"
    assert set(ds[0]) == {"prompt", "gold", "premise", "hypothesis"}


def test_sft_missing_teacher_raises():
    from unittest.mock import patch
    with patch("train.common.ARTIFACTS") as mock_art:
        mock_art.__truediv__.return_value.__truediv__.return_value.exists.return_value = False
        with pytest.raises(NotImplementedError):
            build_sft_dataset(ROWS, "nonexistent_fmt")


def test_sft_dataset_align(tmp_path, monkeypatch):
    from common.io import write_jsonl
    monkeypatch.setattr("train.common.ARTIFACTS", tmp_path)
    write_jsonl(tmp_path / "teacher" / "rationales_align.jsonl", [{
        "id": ROWS[0]["id"],
        "completion": '<align>\nH1: "H1" <-> ∅ | missing\n</align><answer>neutral</answer>'
    }])
    ds = build_sft_dataset(ROWS, "align")
    assert "<align>" in ds[0]["completion"][0]["content"]
    assert "<answer>" in ds[0]["completion"][0]["content"]
    assert "uit_Adver" not in str(ds)
    assert "intended" not in str(ds)


def test_patch_torchao_compatibility():
    import sys
    from types import ModuleType
    from train.common import patch_torchao_compatibility

    patch_torchao_compatibility()

    fake_peft = ModuleType("peft")
    fake_import_utils = ModuleType("peft.import_utils")
    def buggy_is_torchao_available():
        raise ImportError("Found an incompatible version of torchao. Found version 0.10.0")
    fake_import_utils.is_torchao_available = buggy_is_torchao_available

    fake_torchao = ModuleType("peft.tuners.lora.torchao")
    fake_torchao.is_torchao_available = buggy_is_torchao_available

    sys.modules["peft"] = fake_peft
    sys.modules["peft.import_utils"] = fake_import_utils
    sys.modules["peft.tuners.lora.torchao"] = fake_torchao
    try:
        patch_torchao_compatibility()
        assert fake_import_utils.is_torchao_available() is False
        assert fake_torchao.is_torchao_available() is False
    finally:
        sys.modules.pop("peft.tuners.lora.torchao", None)
        sys.modules.pop("peft.import_utils", None)
        sys.modules.pop("peft", None)


def test_patch_vllm_sync_weights_for_qwen35():
    import sys
    from types import ModuleType
    from train.common import patch_vllm_sync_weights_for_qwen35

    # Should run cleanly without trl
    patch_vllm_sync_weights_for_qwen35()

    # Mock trl.generation.vllm_generation
    fake_trl = ModuleType("trl")
    fake_gen = ModuleType("trl.generation")
    fake_vllm_gen = ModuleType("trl.generation.vllm_generation")

    class FakeVLLMGeneration:
        def _fix_param_name_to_vllm(self, name: str, extra_prefixes=None):
            return name
        def sync_weights(self):
            raise ValueError("There is no module or parameter named 'model'")

    calls = []
    def fake_llm(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    fake_vllm_gen.VLLMGeneration = FakeVLLMGeneration
    fake_vllm_gen.LLM = fake_llm
    sys.modules["trl"] = fake_trl
    sys.modules["trl.generation"] = fake_gen
    sys.modules["trl.generation.vllm_generation"] = fake_vllm_gen
    try:
        patch_vllm_sync_weights_for_qwen35()
        inst = FakeVLLMGeneration()
        # model.layers should get language_model. prefix
        assert inst._fix_param_name_to_vllm("model.layers.0.weight") == "language_model.model.layers.0.weight"
        # already prefixed should not be duplicated
        assert inst._fix_param_name_to_vllm("language_model.model.layers.0.weight") == "language_model.model.layers.0.weight"
        with pytest.raises(ValueError, match="There is no module"):
            inst.sync_weights()
        inst_llm = fake_vllm_gen.LLM(model="Qwen/Qwen3.5-2B")
        assert inst_llm is not None
        assert calls[-1][1]["language_model_only"] is True
    finally:
        sys.modules.pop("trl.generation.vllm_generation", None)
        sys.modules.pop("trl.generation", None)
        sys.modules.pop("trl", None)
