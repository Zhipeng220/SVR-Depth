"""Utility exports loaded lazily to avoid importing optional training tools."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "invert_pinhole": "camera", "project_pinhole": "camera", "unproject_pinhole": "camera",
    "barrier": "distributed", "get_dist_info": "distributed", "get_rank": "distributed",
    "get_world_size": "distributed", "is_main_process": "distributed",
    "setup_multi_processes": "distributed", "setup_slurm": "distributed",
    "sync_tensor_across_gpus": "distributed",
    "DICT_METRICS": "evaluation_depth", "DICT_METRICS_3D": "evaluation_depth",
    "eval_3d": "evaluation_depth", "eval_depth": "evaluation_depth",
    "spherical_zbuffer_to_euclidean": "geometric", "unproject_points": "geometric",
    "format_seconds": "misc", "get_params": "misc", "identity": "misc",
    "recursive_index": "misc", "remove_padding": "misc", "to_cpu": "misc",
    "validate": "validation",
    "colorize": "visualization", "image_grid": "visualization",
    "log_train_artifacts": "visualization",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
