"""Explicit state used by streaming interventions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class StreamState:
    radius: Any
    confidence: Any
    features: list[Any]
    rays: Any
    # Extra caches must be carried explicitly, not silently discarded.
    extra: dict[str, Any]

    def clone(self) -> 'StreamState':
        def copy(x):
            if hasattr(x, 'detach') and hasattr(x, 'clone'):
                return x.detach().clone()
            if isinstance(x, dict):
                return {k: copy(v) for k, v in x.items()}
            if isinstance(x, list):
                return [copy(v) for v in x]
            if isinstance(x, tuple):
                return tuple(copy(v) for v in x)
            if isinstance(x, (str, int, float, bool, type(None))):
                return x
            raise TypeError(f'Unsupported state field: {type(x).__name__}')
        return StreamState(copy(self.radius), copy(self.confidence), copy(self.features),
                           copy(self.rays), copy(self.extra))
