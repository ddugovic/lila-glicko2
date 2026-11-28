import argparse
import os
import time
import numpy as np
from dataset_loader import list_available_partitions, prepare_datasets
from glicko2 import LILA_RATING_PERIOD_DAYS, active_numba_threads, load_encoded_games, sweep_rating_periods
from partitions import CATEGORY_REPOS, STANDARD_SPEEDS, VALID_SPEEDS, expand_partitions, resolve_targets


def run_sweep(
    name: str,
    path: str,
    grid: np.ndarray,
    warmup_days: float | None = None,
    summary_path: str | None = None,
    csv_path: str | None = None,
    top_k: int = 5,
):
    print(f"\n=== Dataset: {name} ({path}) ===")
    t0 = time.perf_counter()
    white_ids, black_ids, scores, days, n_players = load_encoded_games(path)
    print(f"Loaded {len(scores):,} games ({n_players:,} players) in {time.perf_counter() - t0:.2f}s")

    warmup = int(np.searchsorted(days, days + warmup_days)) if warmup_days is not None else min(500_000, len(scores) // 10)
    print(f"Warmup: {warmup:,} games")

    t1 = time.perf_counter()
    log_losses, brier_scores = sweep_rating_periods(white_ids, black_ids, scores, days, n_players, grid, warmup_games=warmup)
    elapsed = time.perf_counter() - t1

    ranked_indices = np.argsort(log_losses)
    best_idx = int(np.argmin(log_losses))
    best_bs_idx = int(np.argmin(brier_scores))
    lila_idx = int(np.argmin(np.abs(grid - LILA_RATING_PERIOD_DAYS)))

    print(f"Swept {len(grid)} values in {elapsed:.2f}s ({elapsed / len(grid):.3f}s/run)")
    for rank, idx in enumerate(ranked_indices[:top_k], 1):
        tag = " [Lila]" if idx == lila_idx else ""
        print(f"  #{rank}: period={grid[idx]:.3f}d -> log_loss={log_losses[idx]:.6f}, brier={brier_scores[idx]:.6f}{tag}")
    print(f"  Best Brier: period={grid[best_bs_idx]:.3f}d -> log_loss={log_losses[best_bs_idx]:.6f}, brier={brier_scores[best_bs_idx]:.6f}")
    if lila_idx not in ranked_indices[:top_k]:
        print(f"  Lila Base : period={grid[lila_idx]:.3f}d -> log_loss={log_losses[lila_idx]:.6f}, brier={brier_scores[lila_idx]:.6f}")

    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("dataset,period_days,log_loss,brier_score\n")
            for p, ll, bs in zip(grid, log_losses, brier_scores):
                f.write(f"{name},{p:.6f},{ll:.8f},{bs:.8f}\n")

    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"### Glicko-2 Sweep: `{name}` ({len(scores):,} games, {n_players:,} players, warmup={warmup:,})\n\n")
            f.write("| Rank / Benchmark | Rating Period (days) | Log-Loss | Brier | Δ Log-Loss vs. Best |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            best_ll = log_losses[best_idx]
            for rank, idx in enumerate(ranked_indices[:top_k], 1):
                label = f"**#{rank}**" + (" (Lila Baseline)" if idx == lila_idx else "")
                delta = log_losses[idx] - best_ll
                f.write(f"| {label} | `{grid[idx]:.3f}d` | `{log_losses[idx]:.6f}` | `{brier_scores[idx]:.6f}` | `+{delta:.6f}` |\n")
            if best_bs_idx not in ranked_indices[:top_k]:
                delta = log_losses[best_bs_idx] - best_ll
                f.write(f"| **Best Brier** | `{grid[best_bs_idx]:.3f}d` | `{log_losses[best_bs_idx]:.6f}` | `{brier_scores[best_bs_idx]:.6f}` | `+{delta:.6f}` |\n")
            if lila_idx not in ranked_indices[:top_k]:
                delta = log_losses[lila_idx] - best_ll
                f.write(f"| **Lila Baseline** | `{grid[lila_idx]:.3f}d` | `{log_losses[lila_idx]:.6f}` | `{brier_scores[lila_idx]:.6f}` | `+{delta:.6f}` |\n")
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and stitch Lichess categories/speeds for Glicko-2 sweep.")
    parser.add_argument("--category", type=str, nargs="+", default=["standard"], help=f"all, variants, {', '.join([*CATEGORY_REPOS, *STANDARD_SPEEDS])}")
    parser.add_argument("--speed", type=str, nargs="+", default=None, help=f"all, {', '.join(VALID_SPEEDS)}")
    parser.add_argument("--partitions", type=str, nargs="+", default=["2013"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--warmup-days", type=float, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--list-partitions", action="store_true")
    parser.add_argument("--summary", type=str, default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args()

    targets = resolve_targets(args.category, args.speed)
    if args.list_partitions:
        list_available_partitions(targets, hf_token=os.environ.get("HF_TOKEN"))
        raise SystemExit(0)

    partitions = expand_partitions(args.partitions)
    print(f"Numba parallel worker threads: {active_numba_threads()}")

    first_part = next(iter(partitions))
    part_tag = first_part if len(partitions) == 1 else f"{first_part}-to-{partitions[-1]}"
    date_tag = f"_{args.start_date or 'start'}_to_{args.end_date or 'end'}" if (args.start_date or args.end_date) else ""

    grid = np.unique(np.concatenate([
        np.linspace(0.75, 2.25, 7),
        np.array([2.5, 3.0, 3.5, 4.0, LILA_RATING_PERIOD_DAYS, 5.0, 5.5, 6.0, 8.0, 10.0]),
    ]))

    try:
        datasets = prepare_datasets(
            targets=targets,
            partitions=partitions,
            part_tag=part_tag,
            date_tag=date_tag,
            start_date=args.start_date,
            end_date=args.end_date,
            refresh=args.refresh,
            hf_token=os.environ.get("HF_TOKEN"),
        )
        for target in targets:
            if target in datasets:
                label, parquet_path = datasets[target]
                run_sweep(
                    label,
                    parquet_path,
                    grid,
                    warmup_days=args.warmup_days,
                    summary_path=args.summary,
                    csv_path=f"{label}_sweep.csv",
                )
    except KeyboardInterrupt:
        print("\nInterrupted; keeping results from completed sweeps.")

