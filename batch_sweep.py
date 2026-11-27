import argparse
import os
import time
import numpy as np
from dataset_loader import CATEGORIES, prepare_dataset, resolve_categories
from glicko2 import LILA_RATING_PERIOD_DAYS, active_numba_threads, load_encoded_games, sweep_rating_periods


def run_sweep(name: str, path: str, grid: np.ndarray, warmup_days: float | None = None, summary_path: str | None = None, csv_path: str | None = None):
    print(f"\n=== Dataset: {name} ({path}) ===")
    t0 = time.perf_counter()
    white_ids, black_ids, scores, days, n_players = load_encoded_games(path)
    print(f"Loaded {len(scores):,} games ({n_players:,} players) in {time.perf_counter() - t0:.2f}s")

    warmup = int(np.searchsorted(days, days + warmup_days)) if warmup_days is not None else min(500_000, len(scores) // 10)
    print(f"Warmup: {warmup:,} games")

    t1 = time.perf_counter()
    log_losses, brier_scores = sweep_rating_periods(white_ids, black_ids, scores, days, n_players, grid, warmup_games=warmup)
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

    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"### Glicko-2 Sweep: `{name}` ({len(scores):,} games, {n_players:,} players, warmup={warmup:,})\n\n")
            f.write("| Metric | Rating Period (days) | Value |\n| :--- | :---: | :---: |\n")
            f.write(f"| **Best Log-Loss** | `{grid[best_ll_idx]:.3f}d` | `{log_losses[best_ll_idx]:.6f}` |\n")
            f.write(f"| **Best Brier** | `{grid[best_bs_idx]:.3f}d` | `{brier_scores[best_bs_idx]:.6f}` |\n")
            f.write(f"| **Lila Baseline (Log-Loss)** | `{grid[lila_idx]:.3f}d` | `{log_losses[lila_idx]:.6f}` |\n")
            f.write(f"| **Lila Baseline (Brier)** | `{grid[lila_idx]:.3f}d` | `{brier_scores[lila_idx]:.6f}` |\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and stitch Lichess categories for Glicko-2 rating period sweep.")
    parser.add_argument(
        "--category",
        type=str,
        nargs="+",
        default=["classical"],
        help=f"One or more rating categories, or 'all' ({', '.join(['all', *CATEGORIES.keys()])})",
    )
    parser.add_argument("--partitions", type=str, nargs="+", default=["2013-01"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--warmup-days", type=float, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--summary", type=str, default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args()

    categories = resolve_categories(args.category)
    print(f"Numba parallel worker threads: {active_numba_threads()}")

    first_part = next(iter(args.partitions))
    part_tag = first_part if len(args.partitions) == 1 else f"{first_part}-to-{args.partitions[-1]}"
    date_tag = f"_{args.start_date or 'start'}_to_{args.end_date or 'end'}" if (args.start_date or args.end_date) else ""

    grid = np.unique(np.concatenate([
        np.linspace(0.75, 2.25, 7),
        np.array([2.5, 3.0, 3.5, 4.0, LILA_RATING_PERIOD_DAYS, 5.0, 5.5, 6.0]),
    ]))

    for cat in categories:
        label = f"{cat.replace('-', '_')}_{part_tag}{date_tag}"
        parquet_path = prepare_dataset(
            category=cat,
            partitions=args.partitions,
            label=label,
            start_date=args.start_date,
            end_date=args.end_date,
            refresh=args.refresh,
            hf_token=os.environ.get("HF_TOKEN"),
        )
        if parquet_path:
            run_sweep(label, parquet_path, grid, warmup_days=args.warmup_days, summary_path=args.summary, csv_path=f"{label}_sweep.csv")

