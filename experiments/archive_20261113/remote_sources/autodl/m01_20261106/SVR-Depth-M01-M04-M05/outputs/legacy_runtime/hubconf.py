"""Torch Hub entrypoint for the upstream VeloDepth model."""

import json

import huggingface_hub
from safetensors.torch import load_file

from velodepth.models.velodepth import VeloDepth as _VeloDepth


def VeloDepth(pretrained: bool = True):
    """Build VeloDepth from the official Hugging Face snapshot.

    ``pretrained=False`` creates the same architecture with random weights.
    """
    config_path = huggingface_hub.hf_hub_download(
        repo_id="lpiccinelli/velodepth", filename="config.json", repo_type="model"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    model = _VeloDepth(config)
    if pretrained:
        weights_path = huggingface_hub.hf_hub_download(
            repo_id="lpiccinelli/velodepth",
            filename="model.safetensors",
            repo_type="model",
        )
        state = load_file(weights_path, device="cpu")
        model.load_state_dict(state, strict=True)
    return model
