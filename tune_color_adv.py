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


def _game_ll(eval_fn, adv_elo: float) -> float:
    game_ll, _, _, _ = eval_fn(adv_elo)
    return game_ll


def _est_ll(eval_fn, adv_elo: float) -> float:
    _, _, est_ll, _ = eval_fn(adv_elo)
    return est_ll


def _rank_ll(eval_fn, adv_elo: float) -> float:
    _, _, _, rank_ll = eval_fn(adv_elo)
    return rank_ll


def optimize_color_advantage(
    name: str,
    parquet_path: str,
    rankable_rd: float,
    warmup_days: float | None = None,
    summary_path: str | None = None,
):
    white_ids, black_ids, scores, days, n_players = load_encoded_games(parquet_path)
    warmup = int(days.searchsorted(days + warmup_days)) if warmup_days is not None else min(500_000, len(scores) // 10)
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
        (game_ll, game_brier, est_ll, _, rank_ll, *_), _ = evaluate_rating_period(
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
        return float(game_ll), float(game_brier), float(est_ll), float(rank_ll)

    t0 = time.perf_counter()
    ll_zero, bs_zero, _, _ = eval_adv(0.0)

    res_all = minimize_scalar(lambda x: _game_ll(eval_adv, x), bounds=(-100.0, 200.0), method="bounded", options={"xatol": 1e-6})
    best_all = float(res_all.x)
    ll_best, bs_best, _, _ = eval_adv(best_all)

    best_est = float("nan")
    if est_n > 0:
        res_est = minimize_scalar(lambda x: _est_ll(eval_adv, x), bounds=(-100.0, 200.0), method="bounded", options={"xatol": 1e-6})
        best_est = float(res_est.x)

    best_rank = float("nan")
    if rank_n > 0:
        res_rank = minimize_scalar(lambda x: _rank_ll(eval_adv, x), bounds=(-100.0, 200.0), method="bounded", options={"xatol": 1e-6})
        best_rank = float(res_rank.x)

    elapsed = time.perf_counter() - t0
    print(
        f"{name:28s} | All={best_all:9.6f}d | Est(n={est_n})={best_est:9.6f}d | Rank(n={rank_n})={best_rank:9.6f}d "
        f"| LL: {ll_zero:.6f} -> {ll_best:.6f} ({elapsed:.2f}s)"
    )

    if summary_path:
        est_str = f"`{best_est:.6f}d` ({est_n:,})" if est_n > 0 else "—"
        rank_str = f"`{best_rank:.6f}d` ({rank_n:,})" if rank_n > 0 else "—"
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                f"| `{name}` | {eval_games:,} | `{best_all:.6f}d` | {est_str} | {rank_str} | "
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
    first_part, *_, last_part = partitions if len(partitions) > 1 else (partitions.pop(), partitions[-1])
    part_tag = first_part if first_part == last_part else f"{first_part}-to-{last_part}"

    if args.summary:
        with open(summary_path := args.summary, "a", encoding="utf-8") as f:
            f.write("## Glicko-2 ColorAdvantage Derivation\n\n")
            f.write(
                "| Dataset | Eval Games | All Games `ColorAdvantage` | Established `ColorAdvantage` | "
                "Rankable `ColorAdvantage` | Log-Loss (`0` -> Optimal) | Brier (`0` -> Optimal) |\n"
            )
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

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
