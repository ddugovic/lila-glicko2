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
    PROVISIONAL_RD,
    STANDARD_COLOR_ADV_ELO,
    STANDARD_RANKABLE_RD,
    TAU,
    VARIANT_RANKABLE_RD,
    age_phi,
    expected_score,
    update_player_m1,
)


@njit(fastmath=True, cache=True)
def compute_reference_rds(
    white_ids: np.ndarray, black_ids: np.ndarray, scores: np.ndarray, days: np.ndarray,
    n_players: int, white_adv_elo: float = STANDARD_COLOR_ADV_ELO, tau: float = TAU,
):
    """Run Lila baseline (4.665d) once to record fixed pre-game RDs (float32) for locked cohorts."""
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

    for i in range(n_games):
        w, b, s_w, t = white_ids[i], black_ids[i], scores[i], days[i]
        dt_w = (t - last_played[w]) if last_played[w] >= 0.0 else -1.0
        dt_b = (t - last_played[b]) if last_played[b] >= 0.0 else -1.0

        phi_star_w = age_phi(phis[w], sigmas[w], dt_w, LILA_RATING_PERIOD_DAYS, max_phi) if dt_w >= 0.0 else max_phi
        phi_star_b = age_phi(phis[b], sigmas[b], dt_b, LILA_RATING_PERIOD_DAYS, max_phi) if dt_b >= 0.0 else max_phi

        ref_rd_w[i] = np.float32(phi_star_w * GLICKO2_SCALE)
        ref_rd_b[i] = np.float32(phi_star_b * GLICKO2_SCALE)

        mus[w], phis[w], sigmas[w] = update_player_m1(
            mus[w], phi_star_w, sigmas[w], mus[b] - white_adv_mu, phi_star_b, s_w, tau, min_phi, min_mu, max_mu
        )
        mus[b], phis[b], sigmas[b] = update_player_m1(
            mus[b], phi_star_b, sigmas[b], mus[w] + white_adv_mu, phi_star_w, 1.0 - s_w, tau, min_phi, min_mu, max_mu
        )
        last_played[w], last_played[b] = t, t

    return ref_rd_w, ref_rd_b


@njit(fastmath=True, cache=True)
def _macro_avg(ll_sum: np.ndarray, bs_sum: np.ndarray, counts: np.ndarray):
    total_ll, total_bs, n_active = 0.0, 0.0, 0
    for u in range(len(counts)):
        c = counts[u]
        if c > 0:
            total_ll += ll_sum[u] / c
            total_bs += bs_sum[u] / c
            n_active += 1
    if n_active == 0:
        return np.nan, np.nan, 0
    return total_ll / n_active, total_bs / n_active, n_active


@njit(fastmath=True, cache=True)
def evaluate_rating_period(
    white_ids: np.ndarray, black_ids: np.ndarray, scores: np.ndarray, days: np.ndarray,
    ref_rd_w: np.ndarray, ref_rd_b: np.ndarray, n_players: int,
    rating_period_days: float = LILA_RATING_PERIOD_DAYS,
    rankable_rd: float = STANDARD_RANKABLE_RD,
    white_adv_elo: float = STANDARD_COLOR_ADV_ELO,
    tau: float = TAU, warmup_games: int = 500_000, min_idle_days: float = 0.0,
):
    """Return 16 metrics (8 LL/Brier pairs) and 8 counts covering game, all-player, unlocked, and locked cohorts."""
    min_mu = (MIN_RATING - INIT_RATING) / GLICKO2_SCALE
    max_mu = (MAX_RATING - INIT_RATING) / GLICKO2_SCALE
    max_phi = MAX_RD / GLICKO2_SCALE
    min_phi = MIN_RD / GLICKO2_SCALE
    white_adv_mu = white_adv_elo / GLICKO2_SCALE

    mus = np.zeros(n_players, dtype=np.float64)
    phis = np.full(n_players, max_phi, dtype=np.float64)
    sigmas = np.full(n_players, INIT_VOL, dtype=np.float64)
    last_played = np.full(n_players, -1.0, dtype=np.float64)

    # 0: all players, 1..3: unlocked (prov, est, rank), 4..6: locked (prov, est, rank)
    ll_sums = np.zeros((7, n_players), dtype=np.float64)
    bs_sums = np.zeros((7, n_players), dtype=np.float64)
    cnts = np.zeros((7, n_players), dtype=np.int32)

    total_log_loss, total_brier, eval_count = 0.0, 0.0, 0

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
                p = min(max(e_pred, 1e-12), 1.0 - 1e-12)
                g_ll = -(s_w * math.log(p) + (1.0 - s_w) * math.log(1.0 - p))
                err = s_w - e_pred
                g_bs = err * err

                total_log_loss += g_ll
                total_brier += g_bs
                eval_count += 1

                for side in range(2):
                    u = w if side == 0 else b
                    rd_dyn = (phi_star_w if side == 0 else phi_star_b) * GLICKO2_SCALE
                    rd_fix = float(ref_rd_w[i] if side == 0 else ref_rd_b[i])

                    ll_sums[0, u] += g_ll
                    bs_sums[0, u] += g_bs
                    cnts[0, u] += 1

                    # Unlocked cohorts (slots 1..3)
                    if rd_dyn >= PROVISIONAL_RD:
                        ll_sums[1, u] += g_ll
                        bs_sums[1, u] += g_bs
                        cnts[1, u] += 1
                    else:
                        ll_sums[2, u] += g_ll
                        bs_sums[2, u] += g_bs
                        cnts[2, u] += 1
                        if rd_dyn <= rankable_rd:
                            ll_sums[3, u] += g_ll
                            bs_sums[3, u] += g_bs
                            cnts[3, u] += 1

                    # Locked baseline cohorts (slots 4..6)
                    if rd_fix >= PROVISIONAL_RD:
                        ll_sums[4, u] += g_ll
                        bs_sums[4, u] += g_bs
                        cnts[4, u] += 1
                    else:
                        ll_sums[5, u] += g_ll
                        bs_sums[5, u] += g_bs
                        cnts[5, u] += 1
                        if rd_fix <= rankable_rd:
                            ll_sums[6, u] += g_ll
                            bs_sums[6, u] += g_bs
                            cnts[6, u] += 1

        mus[w], phis[w], sigmas[w] = update_player_m1(
            mus[w], phi_star_w, sigmas[w], mus[b] - white_adv_mu, phi_star_b, s_w, tau, min_phi, min_mu, max_mu
        )
        mus[b], phis[b], sigmas[b] = update_player_m1(
            mus[b], phi_star_b, sigmas[b], mus[w] + white_adv_mu, phi_star_w, 1.0 - s_w, tau, min_phi, min_mu, max_mu
        )
        last_played[w], last_played[b] = t, t

    out_m = np.empty(16, dtype=np.float64)
    out_c = np.empty(8, dtype=np.int64)
    out_m[0] = (total_log_loss / eval_count) if eval_count > 0 else np.nan
    out_m[1] = (total_brier / eval_count) if eval_count > 0 else np.nan
    out_c[0] = eval_count

    for k in range(7):
        m_ll, m_bs, n_act = _macro_avg(ll_sums[k], bs_sums[k], cnts[k])
        out_m[2 + 2 * k] = m_ll
        out_m[3 + 2 * k] = m_bs
        out_c[1 + k] = n_act

    return out_m, out_c

