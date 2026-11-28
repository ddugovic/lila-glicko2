import math
import numpy as np
from glicko2 import LILA_RATING_PERIOD_DAYS


def delta_log_loss_to_elo(delta_ll: float) -> float:
    """Converts excess log-loss (nats/game) into equivalent RMS Elo rating error."""
    if delta_ll <= 0.0 or math.isnan(delta_ll):
        return 0.0
    scale = 400.0 / math.log(10.0)
    return scale * math.sqrt(2.0 * delta_ll)


def build_cohorts():
    return [
        ("Game-Weighted (All Games)", 0, 1, 0, None, ""),
        ("Player-Weighted (All Players)", 2, 3, 1, None, ""),
        ("Player-Weighted: Provisional", 4, 5, 2, 16, "Bottom"),
        ("Player-Weighted: Established", 6, 7, 3, 16, "Top"),
        ("Player-Weighted: Rankable", 8, 9, 4, 17, "Top"),
        ("Player-Weighted: Provisional", 10, 11, 5, None, "Locked"),
        ("Player-Weighted: Established", 12, 13, 6, None, "Locked"),
        ("Player-Weighted: Rankable", 14, 15, 7, None, "Locked"),
    ]


def write_ranked_table(
    f,
    title: str,
    n_units: int,
    unit_label: str,
    grid: np.ndarray,
    log_losses: np.ndarray,
    brier_scores: np.ndarray,
    lila_idx: int,
    top_k: int,
    thresholds: np.ndarray | None = None,
    scope: str = "",
):
    if np.all(np.isnan(log_losses)) or n_units == 0:
        return
    ranked = np.argsort(log_losses)
    best_idx = int(np.nanargmin(log_losses))
    best_bs_idx = int(np.nanargmin(brier_scores))
    best_ll = float(log_losses[best_idx])

    count_str = f"{scope} {n_units:,} {unit_label}" if scope else f"{n_units:,} {unit_label}"
    f.write(f"#### {title} ({count_str})\n\n")
    if thresholds is not None:
        f.write("| Rank / Benchmark | Rating Period (days) | RD Cutoff | Log-Loss | Brier | Δ Elo vs. Best |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
    else:
        f.write("| Rank / Benchmark | Rating Period (days) | Log-Loss | Brier | Δ Elo vs. Best |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")

    def _row(label: str, idx: int) -> str:
        d_elo = delta_log_loss_to_elo(float(log_losses[idx]) - best_ll)
        if thresholds is not None:
            return (
                f"| {label} | `{grid[idx]:.3f}d` | `{thresholds[idx]:.1f}` | "
                f"`{log_losses[idx]:.6f}` | `{brier_scores[idx]:.6f}` | `+{d_elo:.2f} Elo` |\n"
            )
        return (
            f"| {label} | `{grid[idx]:.3f}d` | `{log_losses[idx]:.6f}` | "
            f"`{brier_scores[idx]:.6f}` | `+{d_elo:.2f} Elo` |\n"
        )

    for rank, idx in enumerate(ranked[:top_k], 1):
        label = f"**#{rank}**" + (" (Lila Baseline)" if idx == lila_idx else "")
        f.write(_row(label, int(idx)))
    if best_bs_idx not in ranked[:top_k]:
        f.write(_row("**Best Brier**", best_bs_idx))
    if lila_idx not in ranked[:top_k]:
        f.write(_row("**Lila Baseline**", lila_idx))
    f.write("\n")


def report_sweep_results(
    name: str,
    grid: np.ndarray,
    metrics: np.ndarray,
    counts: np.ndarray,
    rankable_rd: float,
    white_adv_elo: float,
    warmup: int,
    min_idle_days: float,
    elapsed: float,
    summary_path: str | None = None,
    csv_path: str | None = None,
    top_k: int = 5,
):
    lila_idx = int(np.argmin(np.abs(grid - LILA_RATING_PERIOD_DAYS)))
    cohorts = build_cohorts()

    print(f"Swept {len(grid)} values in {elapsed:.2f}s ({elapsed / len(grid):.3f}s/run)")
    for title, ll_col, bs_col, cnt_col, thresh_col, scope in cohorts:
        ll_arr = metrics[:, ll_col]
        bs_arr = metrics[:, bs_col]
        n_units = int(counts[lila_idx, cnt_col])
        if n_units == 0 or np.all(np.isnan(ll_arr)):
            continue
        best_idx = int(np.nanargmin(ll_arr))
        d_elo = delta_log_loss_to_elo(float(ll_arr[lila_idx] - ll_arr[best_idx]))
        thresh_note = f" [RD cut={metrics[best_idx, thresh_col]:.1f}]" if thresh_col is not None else ""
        scope_note = f" ({scope})" if scope else ""
        print(
            f"  [{title}{scope_note}] n={n_units:,} | #1={grid[best_idx]:.3f}d{thresh_note} "
            f"(LL={ll_arr[best_idx]:.6f}, Brier={bs_arr[best_idx]:.6f}) | Lila (+{d_elo:.2f} Elo)"
        )

    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(
                "dataset,period_days,game_ll,game_brier,player_ll,player_brier,"
                "prov_ll,prov_brier,est_ll,est_brier,rankable_ll,rankable_brier,"
                "prov_locked_ll,prov_locked_brier,est_locked_ll,est_locked_brier,rankable_locked_ll,rankable_locked_brier,"
                "rd_cut_est,rd_cut_rank,"
                "eval_games,all_players,prov_players,est_players,rankable_players,"
                "prov_locked_players,est_locked_players,rankable_locked_players\n"
            )
            for idx, p in enumerate(grid):
                m_str = ",".join(f"{val:.8f}" for val in metrics[idx])
                c_str = ",".join(str(int(val)) for val in counts[idx])
                f.write(f"{name},{p:.6f},{m_str},{c_str}\n")

    if summary_path:
        idle_note = f", min_idle={min_idle_days:.1f}d" if min_idle_days > 0 else ""
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                f"### Glicko-2 Sweep: `{name}` (warmup={warmup:,}, rankable_rd<={rankable_rd:.0f}, "
                f"white_adv=+{white_adv_elo:.2f} Elo{idle_note})\n\n"
            )
            for idx_c, (title, ll_col, bs_col, cnt_col, thresh_col, scope) in enumerate(cohorts):
                unit_label = "games" if idx_c == 0 else "players"
                thresh_arr = metrics[:, thresh_col] if thresh_col is not None else None
                write_ranked_table(
                    f,
                    title,
                    int(counts[lila_idx, cnt_col]),
                    unit_label,
                    grid,
                    metrics[:, ll_col],
                    metrics[:, bs_col],
                    lila_idx,
                    top_k,
                    thresholds=thresh_arr,
                    scope=scope,
                )

