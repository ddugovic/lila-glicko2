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
    PROVISIONAL_RD,
    STANDARD_COLOR_ADV_ELO,
    STANDARD_RANKABLE_RD,
    TAU,
    age_phi,
    update_player_m1,
)


@njit(fastmath=True, cache=True)
def compute_reference_rds(
    white_ids: np.ndarray,
    black_ids: np.ndarray,
    scores: np.ndarray,
    days: np.ndarray,
    n_players: int,
    rankable_rd: float = STANDARD_RANKABLE_RD,
    white_adv_elo: float = STANDARD_COLOR_ADV_ELO,
    tau: float = TAU,
    warmup_games: int = 500_000,
    min_idle_days: float = 0.0,
):
    """Run Lila baseline (4.665d) once to record fixed pre-game RDs and exact Established/Rankable counts."""
    n_games = len(scores)
    ref_rd_w = np.empty(n_games, dtype=np.float32)
    ref_rd_b = np.empty(n_games, dtype=np.float32)

    min_mu = (MIN_RATING - INIT_RATING) / GLICKO2_SCALE
    max_mu = (MAX_RATING - INIT_RATING) / GLICKO2_SCALE
    max_phi = MAX_RD / GLICKO2_SCALE
    min_phi = MIN_RD / GLICKO2_SCALE
    white_adv_mu = white_adv_elo / GLICKO2_SCALE

    mus = np.zeros(n_players, dtype=np.float64)
    phis = np.full(n_players, max_phi, dtype=np.float64)
    sigmas = np.full(n_players, INIT_VOL, dtype=np.float64)
    last_played = np.full(n_players, -1.0, dtype=np.float64)

    ref_min_rd = np.full(n_players, 1e9, dtype=np.float64)
    eval_cnt = np.zeros(n_players, dtype=np.int32)

    for i in range(n_games):
        w, b, s_w, t = white_ids[i], black_ids[i], scores[i], days[i]
        dt_w = (t - last_played[w]) if last_played[w] >= 0.0 else -1.0
        dt_b = (t - last_played[b]) if last_played[b] >= 0.0 else -1.0

        phi_star_w = age_phi(phis[w], sigmas[w], dt_w, LILA_RATING_PERIOD_DAYS, max_phi) if dt_w >= 0.0 else max_phi
        phi_star_b = age_phi(phis[b], sigmas[b], dt_b, LILA_RATING_PERIOD_DAYS, max_phi) if dt_b >= 0.0 else max_phi

        rd_w = phi_star_w * GLICKO2_SCALE
        rd_b = phi_star_b * GLICKO2_SCALE
        ref_rd_w[i] = np.float32(rd_w)
        ref_rd_b[i] = np.float32(rd_b)

        if i >= warmup_games:
            qualifies = True if min_idle_days <= 0.0 else ((dt_w >= min_idle_days) or (dt_b >= min_idle_days))
            if qualifies:
                eval_cnt[w] += 1
                eval_cnt[b] += 1
                if rd_w < ref_min_rd[w]:
                    ref_min_rd[w] = rd_w
                if rd_b < ref_min_rd[b]:
                    ref_min_rd[b] = rd_b

        mus[w], phis[w], sigmas[w] = update_player_m1(mus[w], phi_star_w, sigmas[w], mus[b] - white_adv_mu, phi_star_b, s_w, tau, min_phi, min_mu, max_mu)
        mus[b], phis[b], sigmas[b] = update_player_m1(mus[b], phi_star_b, sigmas[b], mus[w] + white_adv_mu, phi_star_w, 1.0 - s_w, tau, min_phi, min_mu, max_mu)
        last_played[w], last_played[b] = t, t

    est_target_n = 0
    rank_target_n = 0
    for u in range(n_players):
        if eval_cnt[u] > 0:
            m_rd = ref_min_rd[u]
            if m_rd < PROVISIONAL_RD:
                est_target_n += 1
                if m_rd <= rankable_rd:
                    rank_target_n += 1

    return ref_rd_w, ref_rd_b, est_target_n, rank_target_n


@njit(fastmath=True, cache=True)
def _write_gpp_quartiles(out_m: np.ndarray, offset: int, gpp_vals: np.ndarray):
    if len(gpp_vals) == 0:
        return
    q25, q50, q75 = np.quantile(gpp_vals, np.array([0.25, 0.50, 0.75]))
    out_m[offset] = float(q25)
    out_m[offset + 1] = float(q50)
    out_m[offset + 2] = float(q75)


@njit(fastmath=True, cache=True)
def _write_cohort_stats(
    out_m: np.ndarray,
    out_c: np.ndarray,
    m_off: int,
    c_off: int,
    g_off: int,
    w_off: int,
    ll_vals: np.ndarray,
    bs_vals: np.ndarray,
    cnt_vals: np.ndarray,
    w_vals: np.ndarray,
    d_vals: np.ndarray,
    l_vals: np.ndarray,
):
    out_m[m_off] = float(np.mean(ll_vals))
    out_m[m_off + 1] = float(np.mean(bs_vals))
    out_c[c_off] = len(ll_vals)
    out_c[g_off] = int(np.sum(cnt_vals))
    out_c[w_off] = int(np.sum(w_vals))
    out_c[w_off + 1] = int(np.sum(d_vals))
    out_c[w_off + 2] = int(np.sum(l_vals))


@njit(fastmath=True, cache=True)
def summarize_player_cohorts(
    all_ll_sum: np.ndarray,
    all_bs_sum: np.ndarray,
    all_cnt: np.ndarray,
    all_w_cnt: np.ndarray,
    all_d_cnt: np.ndarray,
    all_l_cnt: np.ndarray,
    cand_min_rd: np.ndarray,
    ref_min_rd: np.ndarray,
    total_log_loss: float,
    total_brier: float,
    eval_count: int,
    est_target_n: int,
    rank_target_n: int,
    rankable_rd: float,
    rating_period_days: float,
    eval_span_days: float,
):
    """Build 24 metrics and 22 counts including W D L per cohort."""
    n_players = len(all_cnt)
    n_act = 0
    for u in range(n_players):
        if all_cnt[u] > 0:
            n_act += 1

    out_m = np.full(24, np.nan, dtype=np.float64)
    out_c = np.zeros(22, dtype=np.int64)
    base = 0
    out_m[base] = (total_log_loss / eval_count) if eval_count > 0 else np.nan
    out_m[base + 1] = (total_brier / eval_count) if eval_count > 0 else np.nan
    out_c[base] = eval_count
    out_c[base + 1] = n_act

    if n_act == 0:
        return out_m, out_c

    p_ll = np.empty(n_act, dtype=np.float64)
    p_bs = np.empty(n_act, dtype=np.float64)
    p_cnt = np.empty(n_act, dtype=np.int64)
    p_w = np.empty(n_act, dtype=np.int64)
    p_d = np.empty(n_act, dtype=np.int64)
    p_l = np.empty(n_act, dtype=np.int64)
    p_cand_rd = np.empty(n_act, dtype=np.float64)
    p_ref_rd = np.empty(n_act, dtype=np.float64)
    p_gpp = np.empty(n_act, dtype=np.float64)

    span = eval_span_days if eval_span_days > 1e-6 else 1.0
    j = 0
    for u in range(n_players):
        c = all_cnt[u]
        if c > 0:
            p_ll[j] = all_ll_sum[u] / c
            p_bs[j] = all_bs_sum[u] / c
            p_cnt[j] = c
            p_w[j] = all_w_cnt[u]
            p_d[j] = all_d_cnt[u]
            p_l[j] = all_l_cnt[u]
            p_cand_rd[j] = cand_min_rd[u]
            p_ref_rd[j] = ref_min_rd[u]
            p_gpp[j] = (c / span) * rating_period_days
            j += 1

    order = np.argsort(p_cand_rd)
    n_est = min(est_target_n, n_act)
    n_rank = min(rank_target_n, n_act)

    if n_est > 0:
        est_idx = order[:n_est]
        rd_est_off = 10
        out_m[rd_est_off] = float(p_cand_rd[order[n_est - 1]])
        _write_cohort_stats(out_m, out_c, 2, 2, 6, 10, p_ll[est_idx], p_bs[est_idx], p_cnt[est_idx], p_w[est_idx], p_d[est_idx], p_l[est_idx])
        _write_gpp_quartiles(out_m, 12, p_gpp[est_idx])

    if n_rank > 0:
        rank_idx = order[:n_rank]
        rd_rank_off = 11
        out_m[rd_rank_off] = float(p_cand_rd[order[n_rank - 1]])
        _write_cohort_stats(out_m, out_c, 4, 3, 7, 13, p_ll[rank_idx], p_bs[rank_idx], p_cnt[rank_idx], p_w[rank_idx], p_d[rank_idx], p_l[rank_idx])
        _write_gpp_quartiles(out_m, 15, p_gpp[rank_idx])

    lk_est_mask = p_ref_rd < PROVISIONAL_RD
    if int(np.sum(lk_est_mask)) > 0:
        _write_cohort_stats(out_m, out_c, 6, 4, 8, 16, p_ll[lk_est_mask], p_bs[lk_est_mask], p_cnt[lk_est_mask], p_w[lk_est_mask], p_d[lk_est_mask], p_l[lk_est_mask])
        _write_gpp_quartiles(out_m, 18, p_gpp[lk_est_mask])

    lk_rank_mask = p_ref_rd <= rankable_rd
    if int(np.sum(lk_rank_mask)) > 0:
        _write_cohort_stats(out_m, out_c, 8, 5, 9, 19, p_ll[lk_rank_mask], p_bs[lk_rank_mask], p_cnt[lk_rank_mask], p_w[lk_rank_mask], p_d[lk_rank_mask], p_l[lk_rank_mask])
        _write_gpp_quartiles(out_m, 21, p_gpp[lk_rank_mask])

    return out_m, out_c
