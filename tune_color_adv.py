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


def _pick(eval_fn, idx: int, adv_elo: float) -> float:
    return eval_fn(adv_elo)[idx]


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
        (game_ll, game_brier, est_ll, _, rank_ll, _, est_lk_ll, _, rank_lk_ll, *_), _ = evaluate_rating_period(
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
        return (
            float(game_ll),
            float(game_brier),
            float(est_ll),
            float(rank_ll),
            float(est_lk_ll),
            float(rank_lk_ll),
        )

    t0 = time.perf_counter()
    ll_zero, bs_zero, *_ = eval_adv(0.0)

    res_all = minimize_scalar(lambda x: _pick(eval_adv, 0, x), bounds=(-100.0, 200.0), method="bounded", options={"xatol": 1e-6})
    best_all = float(res_all.x)
    ll_best, bs_best, *_ = eval_adv(best_all)

    best_est = float("nan")
    best_est_lk = float("nan")
    if est_n > 0:
        res_est = minimize_scalar(lambda x: _pick(eval_adv, 2, x), bounds=(-100.0, 200.0), method="bounded", options={"xatol": 1e-6})
        best_est = float(res_est.x)
        res_est_lk = minimize_scalar(lambda x: _pick(eval_adv, 4, x), bounds=(-100.0, 200.0), method="bounded", options={"xatol": 1e-6})
        best_est_lk = float(res_est_lk.x)

    best_rank = float("nan")
    best_rank_lk = float("nan")
    if rank_n > 0:
        res_rank = minimize_scalar(lambda x: _pick(eval_adv, 3, x), bounds=(-100.0, 200.0), method="bounded", options={"xatol": 1e-6})
        best_rank = float(res_rank.x)
        res_rank_lk = minimize_scalar(lambda x: _pick(eval_adv, 5, x), bounds=(-100.0, 200.0), method="bounded", options={"xatol": 1e-6})
        best_rank_lk = float(res_rank_lk.x)

    elapsed = time.perf_counter() - t0
    print(
        f"{name:28s} | All={best_all:9.6f}d | Est(Top/Lk, n={est_n})={best_est:9.6f}d / {best_est_lk:9.6f}d "
        f"| Rank(Top/Lk, n={rank_n})={best_rank:9.6f}d / {best_rank_lk:9.6f}d "
        f"| LL: {ll_zero:.6f} -> {ll_best:.6f} ({elapsed:.2f}s)"
    )

    if summary_path:
        est_str = f"`{best_est:.6f}d` ({est_n:,})" if est_n > 0 else "—"
        rank_str = f"`{best_rank:.6f}d` ({rank_n:,})" if rank_n > 0 else "—"
        est_lk_str = f"`{best_est_lk:.6f}d` ({est_n:,})" if est_n > 0 else "—"
        rank_lk_str = f"`{best_rank_lk:.6f}d` ({rank_n:,})" if rank_n > 0 else "—"
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                f"| `{name}` | {eval_games:,} | `{best_all:.6f}d` | {est_str} | {rank_str} | "
                f"{est_lk_str} | {rank_lk_str} | "
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
    first_part = next(iter(partitions))
    part_tag = first_part if len(partitions) == 1 else f"{first_part}-to-{partitions[-1]}"

    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(f"## Glicko-2 ColorAdvantage Derivation (`{part_tag}`)\n\n")
            f.write(
                "| Dataset | Eval Games | All Games `ColorAdvantage` | Established (Top) | "
                "Rankable (Top) | Established (Locked) | Rankable (Locked) | "
                "Log-Loss (`0` -> Optimal) | Brier (`0` -> Optimal) |\n"
            )
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

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
