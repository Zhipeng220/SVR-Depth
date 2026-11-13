"""Safe model loading helpers for VeloDepth and SVR-Depth repair checkpoints."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .config import configure_persistent_feature_repair

REPAIR_PREFIX = "pixel_decoder.persistent_feature_repair."


def _snapshot_paths(model_source: str | Path, revision: str | None = None) -> Path:
    source = Path(model_source).expanduser()
    if source.is_dir():
        snapshot = source.resolve()
    else:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(
            repo_id=str(model_source),
            revision=revision,
            allow_patterns=["config.json", "model.safetensors"],
        ))
    for filename in ("config.json", "model.safetensors"):
        if not (snapshot / filename).is_file():
            raise FileNotFoundError(f"Model snapshot is missing {filename}: {snapshot}")
    return snapshot


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("repair_head_state_dict", "state_dict", "model_state_dict", "model", "weights"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate:
                payload = candidate
                break
    if not isinstance(payload, dict) or not payload:
        raise TypeError("Checkpoint does not contain a non-empty state dictionary")
    return {str(k): v for k, v in payload.items() if isinstance(v, torch.Tensor)}


def _load_repair_head(model: torch.nn.Module, checkpoint: str | Path) -> None:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if getattr(model.pixel_decoder, "persistent_feature_repair", None) is None:
        raise RuntimeError("Repair checkpoint supplied while repair mode is disabled")

    if path.suffix.lower() == ".safetensors":
        state = load_file(str(path), device="cpu")
    else:
        # weights_only avoids arbitrary pickle object loading.
        state = _unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    # Accept a head-only state dict or a full-model state dict.
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith(REPAIR_PREFIX):
            cleaned[key[len(REPAIR_PREFIX):]] = value
        elif key.startswith("persistent_feature_repair."):
            cleaned[key[len("persistent_feature_repair."):]] = value
        elif key in model.pixel_decoder.persistent_feature_repair.state_dict():
            cleaned[key] = value

    if not cleaned:
        raise RuntimeError(
            "No repair-head parameters found. Expected head-only keys or keys prefixed "
            f"with {REPAIR_PREFIX!r}."
        )
    model.pixel_decoder.persistent_feature_repair.load_state_dict(cleaned, strict=True)


def load_model(
    model_source: str | Path = "lpiccinelli/velodepth",
    *,
    revision: str | None = None,
    repair_mode: str = "off",
    repair_bottleneck: int = 32,
    repair_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Load a VeloDepth snapshot, optionally with a trained repair head.

    If repair is enabled without ``repair_checkpoint``, the zero-initialized
    repair head is an exact identity at initialization; a warning is emitted
    to prevent accidentally treating it as a trained SVR-Depth model.
    """
    if repair_mode not in ("off", "pre_readout", "post_readout"):
        raise ValueError("repair_mode must be off, pre_readout, or post_readout")
    snapshot = _snapshot_paths(model_source, revision=revision)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if repair_mode == "off":
        config = configure_persistent_feature_repair(config, enabled=False)
        if repair_checkpoint is not None:
            raise ValueError("repair_checkpoint requires repair_mode to be enabled")
    else:
        config = configure_persistent_feature_repair(
            config, enabled=True, mode=repair_mode, bottleneck=repair_bottleneck
        )

    from velodepth.models.velodepth import VeloDepth

    model = VeloDepth(config)
    base_state = load_file(str(snapshot / "model.safetensors"), device="cpu")
    result = model.load_state_dict(base_state, strict=(repair_mode == "off"))
    if repair_mode != "off":
        invalid_missing = [k for k in result.missing_keys if not k.startswith(REPAIR_PREFIX)]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(
                "Base VeloDepth checkpoint mismatch: "
                f"missing={invalid_missing[:10]}, unexpected={result.unexpected_keys[:10]}"
            )
        present_repair = [k for k in base_state if k.startswith(REPAIR_PREFIX)]
        missing_repair = [k for k in result.missing_keys if k.startswith(REPAIR_PREFIX)]
        if present_repair and missing_repair and repair_checkpoint is None:
            raise RuntimeError(f"Incomplete repair head in snapshot; missing={missing_repair[:10]}")
        if repair_checkpoint is None:
            if not any(key.startswith(REPAIR_PREFIX) for key in base_state):
                warnings.warn(
                    "Repair mode is enabled but no trained repair checkpoint was supplied; "
                    "the zero-initialized head currently behaves as an identity.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            _load_repair_head(model, repair_checkpoint)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.device(target_device)).eval()
    return model
