from eval.adapters import qwen35_adapter_key


def test_text_only_adapter_matches_native_qwen_checkpoint_namespace():
    src='base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight'
    assert qwen35_adapter_key(src)=='base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.weight'


def test_native_multimodal_and_output_head_names_are_preserved():
    for key in ['base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight','base_model.model.lm_head.lora_A.weight']:
        assert qwen35_adapter_key(key)==key
