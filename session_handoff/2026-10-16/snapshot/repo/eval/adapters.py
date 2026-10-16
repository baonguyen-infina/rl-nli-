"""Export text-only HF LoRA keys into Qwen3.5's native multimodal checkpoint namespace."""
from pathlib import Path
import shutil


def qwen35_adapter_key(key: str) -> str:
    for prefix in ('base_model.model.model.layers.', 'model.layers.'):
        if key.startswith(prefix):
            return prefix.replace('model.layers.', 'model.language_model.layers.') + key[len(prefix):]
    return key


def export_qwen35_adapter(adapter, destination):
    from safetensors.torch import load_file, save_file
    adapter, destination = Path(adapter), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    weights = load_file(str(adapter / 'adapter_model.safetensors'), device='cpu')
    mapped = {qwen35_adapter_key(key): value for key, value in weights.items()}
    if len(mapped) != len(weights):
        raise ValueError('Adapter key remapping produced a collision')
    save_file(mapped, str(destination / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    shutil.copy2(adapter / 'adapter_config.json', destination / 'adapter_config.json')
    count = sum(qwen35_adapter_key(key) != key for key in weights)
    print(f'Qwen3.5 vLLM adapter export: remapped {count}/{len(weights)} tensors', flush=True)
    return destination
