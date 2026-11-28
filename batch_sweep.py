import argparse
import os
import time
import numpy as np
from benchmark import estimate_sweep_seconds, format_duration
from dataset_loader import list_available_partitions, prepare_datasets
from glicko2 import (
    LILA_RATING_PERIOD_DAYS,
    STANDARD_RANKABLE_RD,
    VARIANT_RANKABLE_RD,
    active_numba_threads,
    load_encoded_games,
    sweep_rating_periods,
)
from partitions import (
    CATEGORY_REPOS,
    COLOR_ADV_ELO,
    STANDARD_SPEEDS,
    VALID_SPEEDS,
    expand_partitions,
    resolve_targets,
)
from sweep_report import report_sweep_results


def run_sweep(
    name: str,
    path: str,
    grid: np.ndarray,
    rankable_rd: float = STANDARD_RANKABLE_RD,
    white_adv_elo: float = 0.0,
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

    warmup = (
        int(np.searchsorted(days, days[0] + warmup_days))
        if warmup_days is not None
        else min(500_000, len(scores) // 10)
    )
    est_sec = estimate_sweep_seconds(white_ids, black_ids, scores, days, n_players, len(grid), rankable_rd)
    print(
        f"Warmup: {warmup:,} games | Min idle: {min_idle_days:.2f}d | Rankable RD <= {rankable_rd:.0f} "
        f"| White adv: +{white_adv_elo:.2f} Elo | Est. sweep: ~{format_duration(est_sec)}"
    )

    t1 = time.perf_counter()
    metrics, counts = sweep_rating_periods(
        white_ids,
        black_ids,
        scores,
        days,
        n_players,
        grid,
        rankable_rd=rankable_rd,
        white_adv_elo=white_adv_elo,
        warmup_games=warmup,
        min_idle_days=min_idle_days,
    )
    elapsed = time.perf_counter() - t1

    report_sweep_results(
        name=name,
        grid=grid,
        metrics=metrics,
        counts=counts,
        rankable_rd=rankable_rd,
        white_adv_elo=white_adv_elo,
        warmup=warmup,
        min_idle_days=min_idle_days,
        elapsed=elapsed,
        summary_path=summary_path,
        csv_path=csv_path,
        top_k=top_k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and stitch Lichess categories/speeds for Glicko-2 sweep.")
    parser.add_argument(
        "--category",
        type=str,
        nargs="+",
        default=["standard"],
        help=f"all, variants, {', '.join([*CATEGORY_REPOS, *STANDARD_SPEEDS])}",
    )
    parser.add_argument("--speed", type=str, nargs="+", default=None, help=f"all, {', '.join(VALID_SPEEDS)}")
    parser.add_argument("--partitions", type=str, nargs="+", default=["2013"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--warmup-days", type=float, default=None)
    parser.add_argument("--min-idle-days", type=float, default=0.0)
    parser.add_argument(
        "--white-adv-elo",
        type=float,
        default=None,
        help="Override White advantage in Elo (defaults to COLOR_ADV_ELO map)",
    )
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
        for cat, speed in targets:
            if (cat, speed) in datasets:
                label, parquet_path = datasets[(cat, speed)]
                rankable_rd = STANDARD_RANKABLE_RD if cat == "standard" else VARIANT_RANKABLE_RD
                default_adv = COLOR_ADV_ELO.get(cat, 0.0)
                white_adv = args.white_adv_elo if args.white_adv_elo is not None else default_adv
                run_sweep(
                    label,
                    parquet_path,
                    grid,
                    rankable_rd=rankable_rd,
                    white_adv_elo=white_adv,
                    warmup_days=args.warmup_days,
                    min_idle_days=args.min_idle_days,
                    summary_path=args.summary,
                    csv_path=f"{label}_sweep.csv",
                )
    except KeyboardInterrupt:
        print("\nInterrupted; keeping results from completed sweeps.")

