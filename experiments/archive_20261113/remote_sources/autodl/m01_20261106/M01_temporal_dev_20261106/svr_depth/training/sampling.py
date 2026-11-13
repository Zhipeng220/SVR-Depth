"""Reproducible sampling for multi-sequence training."""
import hashlib
import random

def balanced_order(n_slots: int, steps: int, seed: int) -> list[int]:
    if n_slots < 1 or steps < 0:
        raise ValueError('n_slots must be positive and steps nonnegative')
    rng, order = random.Random(seed), []
    while len(order) < steps:
        block = list(range(n_slots))
        rng.shuffle(block)
        order.extend(block)
    return order[:steps]

def order_sha256(order: list[int]) -> str:
    return hashlib.sha256(','.join(map(str, order)).encode()).hexdigest()
