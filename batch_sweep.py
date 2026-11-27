import argparse
import os
import time
import duckdb
import numpy as np
from huggingface_hub import HfApi

from glicko2 import (
    LILA_RATING_PERIOD_DAYS,
    active_numba_threads,
    load_encoded_games,
    sweep_rating_periods,
)


def stream_dataset(
    repo_id: str,
    year: str,
    month: str,
    speed: str,
    output_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
    hf_token: str | None = None,
) -> str:
    """Streams filtered games from Hugging Face via DuckDB HTTP range requests into a compact Parquet file."""
    api = HfApi(token=hf_token)
    prefix = f"data/year={year}/month={month}/"
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    shard_urls = [
        f"hf://datasets/{repo_id}/{p}"
        for p in repo_files
        if p.startswith(prefix) and p.endswith(".parquet")
    ]

    if not shard_urls:
        raise FileNotFoundError(f"No Parquet shards found under {prefix} in {repo_id}")

    print(f"Found {len(shard_urls)} shard(s) for {year}-{month}. Streaming {speed} games -> {output_path}...")
    t0 = time.perf_counter()

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if hf_token:
        con.execute(f"CREATE SECRET hf_secret (TYPE HUGGINGFACE, TOKEN '{hf_token}');")

    con.execute("SET memory_limit = '10GB';")
    con.execute("SET temp_directory = '/tmp/duckdb_spill';")
    con.execute("PRAGMA enable_progress_bar = true;")
    con.execute("PRAGMA progress_bar_time = 0;")

    where_clauses = [
        "Result IN ('1-0', '0-1', '1/2-1/2')",
        "White IS NOT NULL",
        "Black IS NOT NULL",
        "WhiteRatingDiff IS NOT NULL",
        "BlackRatingDiff IS NOT NULL",
        "UTCDate IS NOT NULL",
        "UTCTime IS NOT NULL",
    ]

    if speed.lower() != "all":
        where_clauses.append(f"Event LIKE '%{speed}%'")
    if start_date:
        where_clauses.append(f"UTCDate >= DATE '{start_date}'")
    if end_date:
        where_clauses.append(f"UTCDate <= DATE '{end_date}'")

    where_sql = " AND ".join(where_clauses)

    con.execute(f"""
    COPY (
        SELECT
            White AS white,
            Black AS black,
            CASE Result
                WHEN '1-0' THEN 1.0
                WHEN '0-1' THEN 0.0
                WHEN '1/2-1/2' THEN 0.5
            END AS score,
            epoch(UTCDate::DATE + UTCTime::TIME) AS timestamp,
            WhiteElo AS white_elo,
            BlackElo AS black_elo,
            TimeControl AS time_control
        FROM read_parquet({shard_urls})
        WHERE {where_sql}
        ORDER BY UTCDate ASC, UTCTime ASC
    ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    summary = con.execute(f"""
        SELECT
            COUNT(*),
            MIN(to_timestamp(timestamp)),
            MAX(to_timestamp(timestamp))
        FROM '{output_path}'
    """).fetchone()

    print(
        f"Streamed {summary[0]:,} games ({summary[1]} to {summary[2]}) "
        f"to {output_path} in {time.perf_counter() - t0:.2f}s"
    )
    return output_path


def run_sweep(name: str, path: str, grid: np.ndarray, summary_path: str | None = None, csv_path: str | None = None):
    print(f"\n=== Dataset: {name} ({path}) ===")
    t0 = time.perf_counter()
    white_ids, black_ids, scores, days, n_players = load_encoded_games(path)
    print(f"Loaded {len(scores):,} games ({n_players:,} players) in {time.perf_counter() - t0:.2f}s")

    # Scale warmup dynamically (e.g., 10% of games, capped at 500k)
    warmup = min(500_000, len(scores) // 10)

    t1 = time.perf_counter()
    log_losses, brier_scores = sweep_rating_periods(
        white_ids, black_ids, scores, days, n_players, grid, warmup_games=warmup
    )
    elapsed = time.perf_counter() - t1

    best_ll_idx = int(np.argmin(log_losses))
    best_bs_idx = int(np.argmin(brier_scores))
    lila_idx = int(np.argmin(np.abs(grid - LILA_RATING_PERIOD_DAYS)))

    print(f"Swept {len(grid)} values in {elapsed:.2f}s ({elapsed / len(grid):.3f}s/run)")
    print(f"Best Log-Loss : period={grid[best_ll_idx]:.3f}d -> {log_losses[best_ll_idx]:.6f}")
    print(f"Best Brier    : period={grid[best_bs_idx]:.3f}d -> {brier_scores[best_bs_idx]:.6f}")
    print(f"Lila Baseline : period={grid[lila_idx]:.3f}d -> log_loss={log_losses[lila_idx]:.6f}, brier={brier_scores[lila_idx]:.6f}")

    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("dataset,period_days,log_loss,brier_score\n")
            for p, ll, bs in zip(grid, log_losses, brier_scores):
                f.write(f"{name},{p:.6f},{ll:.8f},{bs:.8f}\n")
        print(f"Wrote sweep curve CSV to {csv_path}")

    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"### Glicko-2 Sweep: `{name}` ({len(scores):,} games, {n_players:,} players)\n\n")
            f.write("| Metric | Rating Period (days) | Value |\n")
            f.write("| :--- | :---: | :---: |\n")
            f.write(f"| **Best Log-Loss** | `{grid[best_ll_idx]:.3f}d` | `{log_losses[best_ll_idx]:.6f}` |\n")
            f.write(f"| **Best Brier** | `{grid[best_bs_idx]:.3f}d` | `{brier_scores[best_bs_idx]:.6f}` |\n")
            f.write(f"| **Lila Baseline (Log-Loss)** | `{grid[lila_idx]:.3f}d` | `{log_losses[lila_idx]:.6f}` |\n")
            f.write(f"| **Lila Baseline (Brier)** | `{grid[lila_idx]:.3f}d` | `{brier_scores[lila_idx]:.6f}` |\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Lichess games and run Glicko-2 rating period sweep.")
    parser.add_argument("--repo", type=str, default="Lichess/standard-chess-games", help="Hugging Face dataset repo ID")
    parser.add_argument("--year", type=str, default="2015", help="Partition year (e.g. 2015, 2025)")
    parser.add_argument("--month", type=str, default="01", help="Partition month (01-12)")
    parser.add_argument("--speed", type=str, default="Classical", help="Time control filter (Bullet, Blitz, Rapid, Classical, or All)")
    parser.add_argument("--start-date", type=str, default=None, help="Optional inclusive start date (YYYY-MM-DD) for row-group pruning")
    parser.add_argument("--end-date", type=str, default=None, help="Optional inclusive end date (YYYY-MM-DD) for row-group pruning")
    parser.add_argument("--parquet", type=str, default=None, help="Local Parquet cache path (auto-named if omitted)")
    parser.add_argument("--refresh", action="store_true", help="Force re-streaming even if local Parquet cache exists")
    parser.add_argument("--summary", type=str, default=os.environ.get("GITHUB_STEP_SUMMARY"), help="Markdown summary output path")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path for sweep results")
    args = parser.parse_args()

    print(f"Numba parallel worker threads: {active_numba_threads()}")

    date_tag = ""
    if args.start_date or args.end_date:
        date_tag = f"_{args.start_date or 'start'}_to_{args.end_date or 'end'}"

    label = f"{args.speed.lower()}_{args.year}_{args.month}{date_tag}"
    parquet_path = args.parquet or f"{label}_rd_tuning.parquet"

    if args.refresh or not os.path.exists(parquet_path):
        stream_dataset(
            repo_id=args.repo,
            year=args.year,
            month=args.month,
            speed=args.speed,
            output_path=parquet_path,
            start_date=args.start_date,
            end_date=args.end_date,
            hf_token=os.environ.get("HF_TOKEN"),
        )
    else:
        print(f"Using cached Parquet file: {parquet_path}")

    grid = np.unique(
        np.concatenate([
            np.linspace(0.25, 2.0, 15),
            np.array([2.5, 3.0, 3.5, 4.0, LILA_RATING_PERIOD_DAYS, 6.0, 8.0, 10.0]),
        ])
    )

    csv_out = args.csv or f"{label}_sweep.csv"
    run_sweep(label, parquet_path, grid, summary_path=args.summary, csv_path=csv_out)

