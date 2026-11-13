"""VeloDepth model package with SVR-Depth persistent feature repair."""

__all__ = ["VeloDepth"]

def __getattr__(name):
    if name == "VeloDepth":
        from .models.velodepth import VeloDepth
        return VeloDepth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
