import duckdb
import numpy as np
from numba import get_num_threads, njit, prange

from glicko2_calc import (
    STANDARD_COLOR_ADV_ELO,
    STANDARD_RANKABLE_RD,
    TAU,
)
from glicko2_cohorts import compute_reference_rds
from glicko2_math import evaluate_rating_period


def load_encoded_games(parquet_path: str):
    """Factorizes white/black usernames into contiguous 0-based int32 IDs inside DuckDB."""
    con = duckdb.connect()
    con.execute("SET memory_limit = '10GB';")
    con.execute("SET temp_directory = '/tmp/duckdb_spill';")

    con.execute(f"""
        CREATE TEMP TABLE players AS
        SELECT username, (ROW_NUMBER() OVER () - 1)::INTEGER AS id
        FROM (
            SELECT DISTINCT white AS username FROM '{parquet_path}'
            UNION
            SELECT DISTINCT black AS username FROM '{parquet_path}'
        );
    """)
    (n_players,) = con.execute("SELECT COUNT(*) FROM players").fetchone()

    encoded = con.execute(f"""
        SELECT
            pw.id AS white_id,
            pb.id AS black_id,
            g.score::DOUBLE AS score,
            (g.timestamp::DOUBLE / 86400.0) AS day
        FROM '{parquet_path}' g
        JOIN players pw ON g.white = pw.username
        JOIN players pb ON g.black = pb.username
        ORDER BY g.timestamp ASC
    """).fetchnumpy()

    return (
        encoded["white_id"],
        encoded["black_id"],
        encoded["score"],
        encoded["day"],
        int(n_players),
    )


@njit(parallel=True, fastmath=True, cache=True)
def _sweep_parallel(
    white_ids: np.ndarray,
    black_ids: np.ndarray,
    scores: np.ndarray,
    days: np.ndarray,
    ref_rd_w: np.ndarray,
    ref_rd_b: np.ndarray,
    n_players: int,
    est_target_n: int,
    rank_target_n: int,
    period_days_grid: np.ndarray,
    rankable_rd: float,
    white_adv_elo: float,
    tau: float,
    warmup_games: int,
    min_idle_days: float,
):
    n_p = len(period_days_grid)
    metrics = np.empty((n_p, 24), dtype=np.float64)
    counts = np.empty((n_p, 6), dtype=np.int64)

    for idx in prange(n_p):
        m_row, c_row = evaluate_rating_period(
            white_ids,
            black_ids,
            scores,
            days,
            ref_rd_w,
            ref_rd_b,
            n_players,
            est_target_n=est_target_n,
            rank_target_n=rank_target_n,
            rating_period_days=period_days_grid[idx],
            rankable_rd=rankable_rd,
            white_adv_elo=white_adv_elo,
            tau=tau,
            warmup_games=warmup_games,
            min_idle_days=min_idle_days,
        )
        metrics[idx] = m_row
        counts[idx] = c_row

    return metrics, counts


def sweep_rating_periods(
    white_ids: np.ndarray,
    black_ids: np.ndarray,
    scores: np.ndarray,
    days: np.ndarray,
    n_players: int,
    period_days_grid: np.ndarray,
    rankable_rd: float = STANDARD_RANKABLE_RD,
    white_adv_elo: float = STANDARD_COLOR_ADV_ELO,
    tau: float = TAU,
    warmup_games: int = 500_000,
    min_idle_days: float = 0.0,
):
    """Computes baseline reference RDs & target player cohort counts once, then sweeps in parallel."""
    ref_rd_w, ref_rd_b, est_target_n, rank_target_n = compute_reference_rds(
        white_ids,
        black_ids,
        scores,
        days,
        n_players,
        rankable_rd=rankable_rd,
        white_adv_elo=white_adv_elo,
        tau=tau,
        warmup_games=warmup_games,
        min_idle_days=min_idle_days,
    )
    return _sweep_parallel(
        white_ids,
        black_ids,
        scores,
        days,
        ref_rd_w,
        ref_rd_b,
        n_players,
        est_target_n,
        rank_target_n,
        period_days_grid,
        rankable_rd,
        white_adv_elo,
        tau,
        warmup_games,
        min_idle_days,
    )


def active_numba_threads() -> int:
    return get_num_threads()
