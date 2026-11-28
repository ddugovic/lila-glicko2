import argparse
import math
import os
import time
import numpy as np
from dataset_loader import list_available_partitions, prepare_datasets
from glicko2 import LILA_RATING_PERIOD_DAYS, active_numba_threads, load_encoded_games, sweep_rating_periods
from partitions import CATEGORY_REPOS, STANDARD_SPEEDS, VALID_SPEEDS, expand_partitions, resolve_targets


def delta_log_loss_to_elo(delta_ll: float) -> float:
    """Converts excess log-loss (nats/game) into equivalent RMS Elo rating error."""
    if delta_ll <= 0.0 or math.isnan(delta_ll):
        return 0.0
    scale = 400.0 / math.log(10.0)
    return scale * math.sqrt(2.0 * delta_ll)


def _write_ranked_table(f, title: str, grid: np.ndarray, log_losses: np.ndarray, brier_scores: np.ndarray, lila_idx: int, top_k: int):
    ranked = np.argsort(log_losses)
    best_idx = int(np.argmin(log_losses))
    best_bs_idx = int(np.argmin(brier_scores))
    best_ll = float(log_losses[best_idx])

    f.write(f"#### {title}\n\n")
    f.write("| Rank / Benchmark | Rating Period (days) | Log-Loss | Brier | Δ Elo vs. Best |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: |\n")
    for rank, idx in enumerate(ranked[:top_k], 1):
        label = f"**#{rank}**" + (" (Lila Baseline)" if idx == lila_idx else "")
        d_elo = delta_log_loss_to_elo(float(log_losses[idx]) - best_ll)
        f.write(f"| {label} | `{grid[idx]:.3f}d` | `{log_losses[idx]:.6f}` | `{brier_scores[idx]:.6f}` | `+{d_elo:.2f} Elo` |\n")
    if best_bs_idx not in ranked[:top_k]:
        d_elo = delta_log_loss_to_elo(float(log_losses[best_bs_idx]) - best_ll)
        f.write(f"| **Best Brier** | `{grid[best_bs_idx]:.3f}d` | `{log_losses[best_bs_idx]:.6f}` | `{brier_scores[best_bs_idx]:.6f}` | `+{d_elo:.2f} Elo` |\n")
    if lila_idx not in ranked[:top_k]:
        d_elo = delta_log_loss_to_elo(float(log_losses[lila_idx]) - best_ll)
        f.write(f"| **Lila Baseline** | `{grid[lila_idx]:.3f}d` | `{log_losses[lila_idx]:.6f}` | `{brier_scores[lila_idx]:.6f}` | `+{d_elo:.2f} Elo` |\n")
    f.write("\n")


def run_sweep(
    name: str,
    path: str,
    grid: np.ndarray,
    warmup_days: float | None = None,
    min_idle_days: float = 0.0,
    summary_path: str | None = None,
    csv_path: str | None = None,
    top_k: int = 5,
):
    print(f"\n=== Dataset: {name} ({path}) ===")
    t0 = time.perf_counter()
    white_ids, black_ids, scores, days, n_players = load_encoded_games(path)
    print(f"Loaded {len(scores):,} games ({n_players:,} players) in {time.perf_counter() - t0:.2f}s")

    warmup = int(np.searchsorted(days, days[0] + warmup_days)) if warmup_days is not None else min(500_000, len(scores) // 10)
    print(f"Warmup: {warmup:,} games | Min idle filter: {min_idle_days:.2f}d")

    t1 = time.perf_counter()
    g_ll, g_bs, p_ll, p_bs, eval_g, eval_p = sweep_rating_periods(
        white_ids, black_ids, scores, days, n_players, grid, warmup_games=warmup, min_idle_days=min_idle_days
    )
    elapsed = time.perf_counter() - t1

    lila_idx = int(np.argmin(np.abs(grid - LILA_RATING_PERIOD_DAYS)))
    g_best_idx = int(np.argmin(g_ll))
    p_best_idx = int(np.argmin(p_ll))

    print(f"Swept {len(grid)} values in {elapsed:.2f}s ({elapsed / len(grid):.3f}s/run) | Evaluated {int(eval_g[0]):,} games across {int(eval_p[0]):,} players")
    print(f"  [Game-Weighted]   #1={grid[g_best_idx]:.3f}d (LL={g_ll[g_best_idx]:.6f}, Brier={g_bs[g_best_idx]:.6f}) | Lila 4.665d (+{delta_log_loss_to_elo(float(g_ll[lila_idx] - g_ll[g_best_idx])):.2f} Elo)")
    print(f"  [Player-Weighted] #1={grid[p_best_idx]:.3f}d (LL={p_ll[p_best_idx]:.6f}, Brier={p_bs[p_best_idx]:.6f}) | Lila 4.665d (+{delta_log_loss_to_elo(float(p_ll[lila_idx] - p_ll[p_best_idx])):.2f} Elo)")

    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("dataset,period_days,game_log_loss,game_brier,game_delta_elo,player_log_loss,player_brier,player_delta_elo,eval_games,eval_players\n")
            for p, gl, gb, pl, pb, eg, ep in zip(grid, g_ll, g_bs, p_ll, p_bs, eval_g, eval_p):
                g_elo = delta_log_loss_to_elo(float(gl - g_ll[g_best_idx]))
                p_elo = delta_log_loss_to_elo(float(pl - p_ll[p_best_idx]))
                f.write(f"{name},{p:.6f},{gl:.8f},{gb:.8f},{g_elo:.4f},{pl:.8f},{pb:.8f},{p_elo:.4f},{int(eg)},{int(ep)}\n")

    if summary_path:
        idle_note = f", min_idle={min_idle_days:.1f}d" if min_idle_days > 0 else ""
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"### Glicko-2 Sweep: `{name}` ({int(eval_g[0]):,} eval games, {int(eval_p[0]):,} eval players, warmup={warmup:,}{idle_note})\n\n")
            _write_ranked_table(f, "Game-Weighted (Micro Average)", grid, g_ll, g_bs, lila_idx, top_k)
            _write_ranked_table(f, "Player-Weighted (Macro Average — Equal Weight per Player)", grid, p_ll, p_bs, lila_idx, top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and stitch Lichess categories/speeds for Glicko-2 sweep.")
    parser.add_argument("--category", type=str, nargs="+", default=["standard"], help=f"all, variants, {', '.join([*CATEGORY_REPOS, *STANDARD_SPEEDS])}")
    parser.add_argument("--speed", type=str, nargs="+", default=None, help=f"all, {', '.join(VALID_SPEEDS)}")
    parser.add_argument("--partitions", type=str, nargs="+", default=["2013"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--warmup-days", type=float, default=None)
    parser.add_argument("--min-idle-days", type=float, default=0.0, help="Only score games where at least one player was idle >= N days")
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

    grid = np.unique(np.r_[0.5:4.26:0.25, LILA_RATING_PERIOD_DAYS, 5.0:10.01:0.5])

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
                    min_idle_days=args.min_idle_days,
                    summary_path=args.summary,
                    csv_path=f"{label}_sweep.csv",
                )
    except KeyboardInterrupt:
        print("\nInterrupted; keeping results from completed sweeps.")

