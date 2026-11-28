import duckdb
import numpy as np
from numba import get_num_threads, njit, prange
from glicko2_math import LILA_RATING_PERIOD_DAYS, TAU, evaluate_rating_period


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
def sweep_rating_periods(
    white_ids: np.ndarray,
    black_ids: np.ndarray,
    scores: np.ndarray,
    days: np.ndarray,
    n_players: int,
    period_days_grid: np.ndarray,
    tau: float = TAU,
    warmup_games: int = 500_000,
    min_idle_days: float = 0.0,
):
    """Evaluate all candidate rating_period_days values in parallel across CPU threads."""
    n_p = len(period_days_grid)
    game_log_losses = np.empty(n_p, dtype=np.float64)
    game_brier_scores = np.empty(n_p, dtype=np.float64)
    player_log_losses = np.empty(n_p, dtype=np.float64)
    player_brier_scores = np.empty(n_p, dtype=np.float64)
    eval_counts = np.empty(n_p, dtype=np.int64)
    eval_players = np.empty(n_p, dtype=np.int64)

    for idx in prange(n_p):
        g_ll, g_bs, p_ll, p_bs, n_eval_g, n_eval_p = evaluate_rating_period(
            white_ids,
            black_ids,
            scores,
            days,
            n_players,
            rating_period_days=period_days_grid[idx],
            tau=tau,
            warmup_games=warmup_games,
            min_idle_days=min_idle_days,
        )
        game_log_losses[idx] = g_ll
        game_brier_scores[idx] = g_bs
        player_log_losses[idx] = p_ll
        player_brier_scores[idx] = p_bs
        eval_counts[idx] = n_eval_g
        eval_players[idx] = n_eval_p

    return game_log_losses, game_brier_scores, player_log_losses, player_brier_scores, eval_counts, eval_players


def active_numba_threads() -> int:
    return get_num_threads()

