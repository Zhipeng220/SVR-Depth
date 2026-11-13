"""Model exports (loaded lazily to keep submodule imports lightweight)."""

__all__ = ["VeloDepth"]

def __getattr__(name):
    if name == "VeloDepth":
        from .velodepth import VeloDepth
        return VeloDepth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
