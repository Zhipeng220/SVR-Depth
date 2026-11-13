"""Configuration helpers for the SVR-Depth persistent repair module."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

REPAIR_MODES = ("pre_readout", "post_readout")


def configure_persistent_feature_repair(
    config: Mapping[str, Any],
    *,
    enabled: bool = True,
    mode: str = "pre_readout",
    bottleneck: int = 32,
) -> dict[str, Any]:
    """Return a copied VeloDepth config with the repair head enabled/disabled.

    ``pre_readout`` repairs propagated features before the current-frame readout;
    ``post_readout`` preserves the current readout and repairs only the state
    carried into subsequent frames. The input mapping is never mutated.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(bottleneck, int) or bottleneck < 1:
        raise ValueError("bottleneck must be a positive integer")
    if enabled and mode not in REPAIR_MODES:
        raise ValueError(f"mode must be one of {REPAIR_MODES}, got {mode!r}")

    result = deepcopy(dict(config))
    model_cfg = result.setdefault("model", {})
    if not isinstance(model_cfg, dict):
        raise TypeError("config['model'] must be a mapping")
    if enabled:
        model_cfg["persistent_feature_repair"] = {
            "enabled": True,
            "mode": mode,
            "bottleneck": bottleneck,
        }
    else:
        model_cfg.pop("persistent_feature_repair", None)
    return result
