import argparse
import os
import time

from scipy.optimize import minimize_scalar

from dataset_loader import prepare_datasets
from glicko2 import load_encoded_games
from glicko2_calc import LILA_RATING_PERIOD_DAYS, STANDARD_RANKABLE_RD, VARIANT_RANKABLE_RD
from glicko2_cohorts import compute_reference_rds
from glicko2_math import evaluate_rating_period
from partitions import CATEGORY_REPOS, STANDARD_SPEEDS, VALID_SPEEDS, expand_partitions, resolve_targets


def optimize_color_advantage(
    name: str,
    parquet_path: str,
    rankable_rd: float,
    warmup_days: float | None = None,
    summary_path: str | None = None,
):
    white_ids, black_ids, scores, days, n_players = load_encoded_games(parquet_path)
    warmup = int(days.searchsorted(days[0] + warmup_days)) if warmup_days is not None else min(500_000, len(scores) // 10)
    eval_games = max(0, len(scores) - warmup)

    ref_rd_w, ref_rd_b, est_n, rank_n = compute_reference_rds(
        white_ids,
        black_ids,
        scores,
        days,
        n_players,
        rankable_rd=rankable_rd,
        white_adv_elo=0.0,
        warmup_games=warmup,
    )

    def eval_adv(adv_elo: float):
        m, _ = evaluate_rating_period(
            white_ids,
            black_ids,
            scores,
            days,
            ref_rd_w,
            ref_rd_b,
            n_players,
            est_target_n=est_n,
            rank_target_n=rank_n,
            rating_period_days=LILA_RATING_PERIOD_DAYS,
            rankable_rd=rankable_rd,
            white_adv_elo=adv_elo,
            warmup_games=warmup,
        )
        return float(m[0]), float(m[1])  # (game_ll, game_brier)

    t0 = time.perf_counter()
    ll_zero, bs_zero = eval_adv(0.0)
    res = minimize_scalar(lambda x: eval_adv(x)[0], bounds=(-100.0, 200.0), method="bounded", options={"xatol": 1e-6})
    best_adv = float(res.x)
    ll_best, bs_best = eval_adv(best_adv)
    elapsed = time.perf_counter() - t0

    print(
        f"{name:28s} | ColorAdvantage = {best_adv:10.6f}d "
        f"| LL: {ll_zero:.6f} -> {ll_best:.6f} | Brier: {bs_zero:.6f} -> {bs_best:.6f} "
        f"({res.nit} evals, {elapsed:.2f}s)"
    )

    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                f"| `{name}` | {eval_games:,} | `{best_adv:.6f}d` | "
                f"`{ll_zero:.6f}` -> `{ll_best:.6f}` | `{bs_zero:.6f}` -> `{bs_best:.6f}` |\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive exact Glicko-2 White ColorAdvantage (Elo) via Brent MLE.")
    parser.add_argument(
        "--category",
        type=str,
        nargs="+",
        default=["variants"],
        help=f"all, variants, {', '.join([*CATEGORY_REPOS, *STANDARD_SPEEDS])}",
    )
    parser.add_argument("--speed", type=str, nargs="+", default=None, help=f"all, {', '.join(VALID_SPEEDS)}")
    parser.add_argument("--partitions", type=str, nargs="+", default=["2018"])
    parser.add_argument("--warmup-days", type=float, default=None)
    parser.add_argument("--summary", type=str, default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args()

    targets = resolve_targets(args.category, args.speed)
    partitions = expand_partitions(args.partitions)
    part_tag = partitions[0] if len(partitions) == 1 else f"{partitions[0]}-to-{partitions[-1]}"

    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write("## Glicko-2 ColorAdvantage Derivation\n\n")
            f.write("| Dataset | Eval Games | Optimal `ColorAdvantage` | Log-Loss (`0` -> Optimal) | Brier (`0` -> Optimal) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")

    datasets = prepare_datasets(
        targets=targets,
        partitions=partitions,
        part_tag=part_tag,
        date_tag="",
        start_date=None,
        end_date=None,
        refresh=False,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    for cat, speed in targets:
        if (cat, speed) in datasets:
            label, path = datasets[(cat, speed)]
            rank_rd = STANDARD_RANKABLE_RD if cat == "standard" else VARIANT_RANKABLE_RD
            optimize_color_advantage(label, path, rank_rd, warmup_days=args.warmup_days, summary_path=args.summary)
