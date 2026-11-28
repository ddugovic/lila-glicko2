import math

import numpy as np
from numba import njit

from glicko2_calc import (
    GLICKO2_SCALE,
    INIT_RATING,
    INIT_VOL,
    LILA_RATING_PERIOD_DAYS,
    MAX_RATING,
    MAX_RD,
    MIN_RATING,
    MIN_RD,
    STANDARD_COLOR_ADV_ELO,
    STANDARD_RANKABLE_RD,
    TAU,
    age_phi,
    expected_score,
    update_player_m1,
)
from glicko2_cohorts import summarize_player_cohorts


@njit(fastmath=True, cache=True)
def evaluate_rating_period(
    white_ids: np.ndarray,
    black_ids: np.ndarray,
    scores: np.ndarray,
    days: np.ndarray,
    ref_rd_w: np.ndarray,
    ref_rd_b: np.ndarray,
    n_players: int,
    est_target_n: int,
    rank_target_n: int,
    rating_period_days: float = LILA_RATING_PERIOD_DAYS,
    rankable_rd: float = STANDARD_RANKABLE_RD,
    white_adv_elo: float = STANDARD_COLOR_ADV_ELO,
    tau: float = TAU,
    warmup_games: int = 500_000,
    min_idle_days: float = 0.0,
):
    """Simulate full game stream for a candidate rating_period_days."""
    min_mu = (MIN_RATING - INIT_RATING) / GLICKO2_SCALE
    max_mu = (MAX_RATING - INIT_RATING) / GLICKO2_SCALE
    max_phi = MAX_RD / GLICKO2_SCALE
    min_phi = MIN_RD / GLICKO2_SCALE
    white_adv_mu = white_adv_elo / GLICKO2_SCALE

    mus = np.zeros(n_players, dtype=np.float64)
    phis = np.full(n_players, max_phi, dtype=np.float64)
    sigmas = np.full(n_players, INIT_VOL, dtype=np.float64)
    last_played = np.full(n_players, -1.0, dtype=np.float64)

    all_ll_sum = np.zeros(n_players, dtype=np.float64)
    all_bs_sum = np.zeros(n_players, dtype=np.float64)
    all_cnt = np.zeros(n_players, dtype=np.int32)
    all_w_cnt = np.zeros(n_players, dtype=np.int32)
    all_d_cnt = np.zeros(n_players, dtype=np.int32)
    all_l_cnt = np.zeros(n_players, dtype=np.int32)
    cand_min_rd = np.full(n_players, 1e9, dtype=np.float64)
    ref_min_rd = np.full(n_players, 1e9, dtype=np.float64)

    total_log_loss, total_brier, eval_count = 0.0, 0.0, 0
    first_eval_day = -1.0
    last_eval_day = -1.0

    for i in range(len(scores)):
        w, b, s_w, t = white_ids[i], black_ids[i], scores[i], days[i]
        dt_w = (t - last_played[w]) if last_played[w] >= 0.0 else -1.0
        dt_b = (t - last_played[b]) if last_played[b] >= 0.0 else -1.0

        phi_star_w = age_phi(phis[w], sigmas[w], dt_w, rating_period_days, max_phi) if dt_w >= 0.0 else max_phi
        phi_star_b = age_phi(phis[b], sigmas[b], dt_b, rating_period_days, max_phi) if dt_b >= 0.0 else max_phi

        phi_comb = math.sqrt(phi_star_w * phi_star_w + phi_star_b * phi_star_b)
        e_pred = expected_score(mus[w] + white_adv_mu, mus[b], phi_comb)

        if i >= warmup_games:
            qualifies = True if min_idle_days <= 0.0 else ((dt_w >= min_idle_days) or (dt_b >= min_idle_days))
            if qualifies:
                if first_eval_day < 0.0:
                    first_eval_day = t
                last_eval_day = t

                p = min(max(e_pred, 1e-12), 1.0 - 1e-12)
                g_ll = -(s_w * math.log(p) + (1.0 - s_w) * math.log(1.0 - p))
                err = s_w - e_pred
                g_bs = err * err

                total_log_loss += g_ll
                total_brier += g_bs
                eval_count += 1

                rd_w = phi_star_w * GLICKO2_SCALE
                rd_b = phi_star_b * GLICKO2_SCALE
                fix_w = float(ref_rd_w[i])
                fix_b = float(ref_rd_b[i])

                all_ll_sum[w] += g_ll
                all_bs_sum[w] += g_bs
                all_cnt[w] += 1
                all_ll_sum[b] += g_ll
                all_bs_sum[b] += g_bs
                all_cnt[b] += 1

                if s_w == 1.0:
                    all_w_cnt[w] += 1
                    all_w_cnt[b] += 1
                elif s_w == 0.5:
                    all_d_cnt[w] += 1
                    all_d_cnt[b] += 1
                else:
                    all_l_cnt[w] += 1
                    all_l_cnt[b] += 1

                if rd_w < cand_min_rd[w]:
                    cand_min_rd[w] = rd_w
                if fix_w < ref_min_rd[w]:
                    ref_min_rd[w] = fix_w
                if rd_b < cand_min_rd[b]:
                    cand_min_rd[b] = rd_b
                if fix_b < ref_min_rd[b]:
                    ref_min_rd[b] = fix_b

        mus[w], phis[w], sigmas[w] = update_player_m1(mus[w], phi_star_w, sigmas[w], mus[b] - white_adv_mu, phi_star_b, s_w, tau, min_phi, min_mu, max_mu)
        mus[b], phis[b], sigmas[b] = update_player_m1(mus[b], phi_star_b, sigmas[b], mus[w] + white_adv_mu, phi_star_w, 1.0 - s_w, tau, min_phi, min_mu, max_mu)
        last_played[w], last_played[b] = t, t

    eval_span = (last_eval_day - first_eval_day) if first_eval_day >= 0.0 else 1.0
    return summarize_player_cohorts(
        all_ll_sum,
        all_bs_sum,
        all_cnt,
        all_w_cnt,
        all_d_cnt,
        all_l_cnt,
        cand_min_rd,
        ref_min_rd,
        total_log_loss,
        total_brier,
        eval_count,
        est_target_n,
        rank_target_n,
        rankable_rd,
        rating_period_days,
        eval_span,
    )
