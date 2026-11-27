import argparse
import os
import time
import duckdb
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from glicko2 import (
    LILA_RATING_PERIOD_DAYS,
    active_numba_threads,
    load_encoded_games,
    sweep_rating_periods,
)


def prepare_dataset(
    repo_id: str,
    year: str,
    months: list[str],
    speed: str | None,
    label: str,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh: bool = False,
    hf_token: str | None = None,
) -> str | None:
    """
    Downloads and extracts one Parquet shard at a time (1 resolver request per shard),
    deleting each raw shard immediately after extraction to stay within CI disk limits.
    """
    out_path = f"{label}_rd_tuning.parquet"
    if not refresh and os.path.exists(out_path):
        print(f"Using cached Parquet file: {out_path}")
        return out_path

    api = HfApi(token=hf_token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    shard_paths: list[str] = []
    for m in months:
        prefix = f"data/year={year}/month={m}/"
        matched = [p for p in repo_files if p.startswith(prefix) and p.endswith(".parquet")]
        if not matched:
            raise FileNotFoundError(f"No Parquet shards found under {prefix} in {repo_id}")
        shard_paths.extend(matched)

    con = duckdb.connect()
    con.execute("SET memory_limit = '10GB';")
    con.execute("SET temp_directory = '/tmp/duckdb_spill';")

    where_clauses = [
        "Result IN ('1-0', '0-1', '1/2-1/2')",
        "White IS NOT NULL",
        "Black IS NOT NULL",
        "WhiteRatingDiff IS NOT NULL",
        "BlackRatingDiff IS NOT NULL",
        "UTCDate IS NOT NULL",
        "UTCTime IS NOT NULL",
    ]
    if speed:
        where_clauses.append(f"Event LIKE '%{speed}%'")
    if start_date:
        where_clauses.append(f"UTCDate >= DATE '{start_date}'")
    if end_date:
        where_clauses.append(f"UTCDate <= DATE '{end_date}'")

    where_sql = " AND ".join(where_clauses)
    pool_desc = speed or repo_id.split("/")[-1]
    print(f"Streaming {len(shard_paths)} shard(s) 1-by-1 for {pool_desc} -> {out_path}...")
    t0 = time.perf_counter()

    compact_parts: list[str] = []
    try:
        for idx, rel_path in enumerate(shard_paths, 1):
            print(f"  [{idx}/{len(shard_paths)}] Downloading {rel_path}...")
            local_raw = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=rel_path,
                token=hf_token,
            )

            part_path = f".tmp_{label}_part_{idx:04d}.parquet"
            try:
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
                    FROM read_parquet('{local_raw}')
                    WHERE {where_sql}
                ) TO '{part_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)
                compact_parts.append(part_path)
            finally:
                # Resolve symlink to the actual blob in HF cache and delete both to free SSD immediately
                real_blob = os.path.realpath(local_raw)
                if os.path.exists(local_raw):
                    os.remove(local_raw)
                if os.path.exists(real_blob):
                    os.remove(real_blob)

        # Merge and sort all compact parts chronologically into the final Parquet file
        con.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet({compact_parts})
            ORDER BY timestamp ASC
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
    finally:
        for p in compact_parts:
            if os.path.exists(p):
                os.remove(p)

    summary = con.execute(f"""
        SELECT
            COUNT(*),
            MIN(to_timestamp(timestamp)),
            MAX(to_timestamp(timestamp))
        FROM '{out_path}'
    """).fetchone()

    if summary[0] == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        hint = " (Lichess introduced Rapid on Dec 2, 2017)" if speed == "Rapid" and int(year) <= 2017 else ""
        print(f"  Skipping {pool_desc}: 0 games matched in {repo_id} (year={year}, months={months}){hint}.")
        return None

    print(
        f"  Saved {summary[0]:,} {pool_desc} games ({summary[1]} to {summary[2]}) "
        f"in {time.perf_counter() - t0:.2f}s"
    )
    return out_path


def run_sweep(
    name: str,
    path: str,
    grid: np.ndarray,
    warmup_days: float | None = None,
    summary_path: str | None = None,
    csv_path: str | None = None,
):
    print(f"\n=== Dataset: {name} ({path}) ===")
    t0 = time.perf_counter()
    white_ids, black_ids, scores, days, n_players = load_encoded_games(path)
    print(f"Loaded {len(scores):,} games ({n_players:,} players) in {time.perf_counter() - t0:.2f}s")

    if warmup_days:
        warmup = int(np.searchsorted(days, days[0] + warmup_days))
        print(f"Using time-based warmup: {warmup_days:.1f} days ({warmup:,} games)")
    else:
        warmup = min(500_000, len(scores) // 10)
        print(f"Using default count-based warmup: {warmup:,} games")

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
            f.write(f"### Glicko-2 Sweep: `{name}` ({len(scores):,} games, {n_players:,} players, warmup={warmup:,})\n\n")
            f.write("| Metric | Rating Period (days) | Value |\n")
            f.write("| :--- | :---: | :---: |\n")
            f.write(f"| **Best Log-Loss** | `{grid[best_ll_idx]:.3f}d` | `{log_losses[best_ll_idx]:.6f}` |\n")
            f.write(f"| **Best Brier** | `{grid[best_bs_idx]:.3f}d` | `{brier_scores[best_bs_idx]:.6f}` |\n")
            f.write(f"| **Lila Baseline (Log-Loss)** | `{grid[lila_idx]:.3f}d` | `{log_losses[lila_idx]:.6f}` |\n")
            f.write(f"| **Lila Baseline (Brier)** | `{grid[lila_idx]:.3f}d` | `{brier_scores[lila_idx]:.6f}` |\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and stitch Lichess games for Glicko-2 rating period sweep.")
    parser.add_argument("--repo", type=str, default="Lichess/standard-chess-games", help="Hugging Face dataset repo ID")
    parser.add_argument("--year", type=str, default="2015", help="Partition year (e.g. 2015, 2025)")
    parser.add_argument("--months", type=str, nargs="+", default=["01"], help="One or more months to stitch (e.g. 01 02 03)")
    parser.add_argument(
        "--speed",
        type=str,
        choices=["Bullet", "Blitz", "Rapid", "Classical"],
        default=None,
        help="Rating pool filter for standard chess (omit for single-pool variants)",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Optional inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Optional inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--warmup-days", type=float, default=None, help="Optional warmup duration in days")
    parser.add_argument("--refresh", action="store_true", help="Force re-extraction even if local Parquet cache exists")
    parser.add_argument("--summary", type=str, default=os.environ.get("GITHUB_STEP_SUMMARY"), help="Markdown summary output path")
    args = parser.parse_args()

    if args.repo == "Lichess/standard-chess-games" and not args.speed:
        parser.error("--speed (Bullet, Blitz, Rapid, or Classical) is required for Lichess/standard-chess-games.")

    print(f"Numba parallel worker threads: {active_numba_threads()}")

    months_padded = [m.zfill(2) for m in args.months]
    month_tag = months_padded[0] if len(months_padded) == 1 else f"{months_padded[0]}-to-{months_padded[-1]}"
    date_tag = f"_{args.start_date or 'start'}_to_{args.end_date or 'end'}" if (args.start_date or args.end_date) else ""
    pool_tag = args.speed.lower() if args.speed else args.repo.split("/")[-1].replace("-chess-games", "").replace("-", "_")
    label = f"{pool_tag}_{args.year}_{month_tag}{date_tag}"

    parquet_path = prepare_dataset(
        repo_id=args.repo,
        year=args.year,
        months=months_padded,
        speed=args.speed,
        label=label,
        start_date=args.start_date,
        end_date=args.end_date,
        refresh=args.refresh,
        hf_token=os.environ.get("HF_TOKEN"),
    )

    if parquet_path:
        grid = np.unique(
            np.concatenate([
                np.linspace(0.75, 2.25, 13),
                np.array([2.5, 3.0, 3.5, 4.0, LILA_RATING_PERIOD_DAYS, 6.0, 8.0, 10.0]),
            ])
        )

        run_sweep(
            label,
            parquet_path,
            grid,
            warmup_days=args.warmup_days,
            summary_path=args.summary,
            csv_path=f"{label}_sweep.csv",
        )

