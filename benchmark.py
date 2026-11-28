import math
import time

import numpy as np

from glicko2 import active_numba_threads
from glicko2_math import evaluate_rating_period


def estimate_sweep_seconds(
    white_ids: np.ndarray,
    black_ids: np.ndarray,
    scores: np.ndarray,
    days: np.ndarray,
    n_players: int,
    n_grid: int,
    rankable_rd: float,
    sample_games: int = 250_000,
    contention_factor: float = 1.45,
) -> float:
    n_total = len(scores)
    sample_n = min(sample_games, n_total)
    if sample_n == 0 or n_grid <= 0:
        return 0.0

    dummy_rd = np.full(sample_n, 150.0, dtype=np.float32)
    warmup_n = min(256, sample_n)
    evaluate_rating_period(
        white_ids[:warmup_n],
        black_ids[:warmup_n],
        scores[:warmup_n],
        days[:warmup_n],
        dummy_rd[:warmup_n],
        dummy_rd[:warmup_n],
        n_players,
        100,
        50,
        rankable_rd=rankable_rd,
        warmup_games=0,
    )

    t0 = time.perf_counter()
    evaluate_rating_period(
        white_ids[:sample_n],
        black_ids[:sample_n],
        scores[:sample_n],
        days[:sample_n],
        dummy_rd,
        dummy_rd,
        n_players,
        100,
        50,
        rankable_rd=rankable_rd,
        warmup_games=0,
    )
    sample_sec = time.perf_counter() - t0

    threads = max(1, active_numba_threads())
    waves = math.ceil(n_grid / threads)
    return sample_sec * (n_total / sample_n) * (waves * contention_factor + 1.0)


def format_duration(seconds: float) -> str:
    if seconds < 90.0:
        return f"{seconds:.1f}s"
    if seconds < 5400.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.2f} hr"
