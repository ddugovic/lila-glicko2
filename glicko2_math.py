import math
import numpy as np
from numba import njit

# Exact constants from lila (lichess)
GLICKO2_SCALE = 173.7178       # 400 / ln(10)
INIT_RATING = 1500.0
MIN_RATING = 400.0
MAX_RATING = 4000.0

MIN_RD = 45.0
MAX_RD = 500.0                 # lila: maxDeviation = 500d
INIT_VOL = 0.09                # lila: defaultVolatility = 0.09d
MAX_VOL = 0.1                  # lila: maxVolatility = 0.1d
TAU = 0.75                     # lila system constant
EPSILON = 1e-6
LILA_RATING_PERIOD_DAYS = 1.0 / 0.21436  # ~4.66505 days


@njit(fastmath=True, cache=True)
def g_func(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


@njit(fastmath=True, cache=True)
def expected_score(mu: float, mu_opp: float, phi_opp: float) -> float:
    return 1.0 / (1.0 + math.exp(-g_func(phi_opp) * (mu - mu_opp)))


@njit(fastmath=True, cache=True)
def clamp_mu(mu: float, min_mu: float, max_mu: float) -> float:
    if mu < min_mu:
        return min_mu
    if mu > max_mu:
        return max_mu
    return mu


@njit(fastmath=True, cache=True)
def age_phi(
    phi: float,
    sigma: float,
    dt_days: float,
    rating_period_days: float,
    max_phi: float,
) -> float:
    """Apply pre-game RD (phi) growth over elapsed fractional rating periods."""
    if dt_days <= 0.0:
        return phi
    periods = dt_days / rating_period_days
    phi_star = math.sqrt(phi * phi + periods * sigma * sigma)
    return max_phi if phi_star > max_phi else phi_star


@njit(fastmath=True, cache=True)
def update_volatility(phi: float, sigma: float, v: float, delta: float, tau: float) -> float:
    """Glickman's Illinois algorithm with lila's maxVolatility = 0.1 cap."""
    a = math.log(sigma * sigma)
    tau2 = tau * tau
    phi2 = phi * phi
    delta2 = delta * delta

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta2 - phi2 - v - ex)
        den = 2.0 * (phi2 + v + ex) ** 2
        return (num / den) - ((x - a) / tau2)

    A = a
    if delta2 > phi2 + v:
        B = math.log(delta2 - phi2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0.0:
            k += 1
        B = a - k * tau

    fA = f(A)
    fB = f(B)

    while abs(B - A) > EPSILON:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0.0:
            A = B
            fA = fB
        else:
            fA *= 0.5
        B = C
        fB = fC

    new_sigma = math.exp(A * 0.5)
    return MAX_VOL if new_sigma > MAX_VOL else new_sigma


@njit(fastmath=True, cache=True)
def update_player_m1(
    mu: float,
    phi_star: float,
    sigma: float,
    mu_opp: float,
    phi_star_opp: float,
    score: float,
    tau: float,
    min_phi: float,
    min_mu: float,
    max_mu: float,
):
    """Single-game (m = 1) Glicko-2 update returning (new_mu, new_phi, new_sigma)."""
    g_opp = g_func(phi_star_opp)
    e = expected_score(mu, mu_opp, phi_star_opp)
    v = 1.0 / (g_opp * g_opp * e * (1.0 - e))
    delta = v * g_opp * (score - e)

    new_sigma = update_volatility(phi_star, sigma, v, delta, tau)
    new_phi = 1.0 / math.sqrt((1.0 / (phi_star * phi_star)) + (1.0 / v))
    if new_phi < min_phi:
        new_phi = min_phi

    new_mu = clamp_mu(mu + new_phi * new_phi * g_opp * (score - e), min_mu, max_mu)
    return new_mu, new_phi, new_sigma


@njit(fastmath=True, cache=True)
def evaluate_rating_period(
    white_ids: np.ndarray,
    black_ids: np.ndarray,
    scores: np.ndarray,
    days: np.ndarray,
    n_players: int,
    rating_period_days: float = LILA_RATING_PERIOD_DAYS,
    tau: float = TAU,
    warmup_games: int = 500_000,
    min_idle_days: float = 0.0,
):
    """Return (game_ll, game_brier, player_macro_ll, player_macro_brier, eval_games, eval_players)."""
    min_mu = (MIN_RATING - INIT_RATING) / GLICKO2_SCALE
    max_mu = (MAX_RATING - INIT_RATING) / GLICKO2_SCALE
    max_phi = MAX_RD / GLICKO2_SCALE
    min_phi = MIN_RD / GLICKO2_SCALE

    mus = np.zeros(n_players, dtype=np.float64)
    phis = np.full(n_players, max_phi, dtype=np.float64)
    sigmas = np.full(n_players, INIT_VOL, dtype=np.float64)
    last_played = np.full(n_players, -1.0, dtype=np.float64)

    player_ll_sum = np.zeros(n_players, dtype=np.float64)
    player_bs_sum = np.zeros(n_players, dtype=np.float64)
    player_eval_games = np.zeros(n_players, dtype=np.int32)

    total_log_loss = 0.0
    total_brier = 0.0
    eval_count = 0

    for i in range(len(scores)):
        w = white_ids[i]
        b = black_ids[i]
        s_w = scores[i]
        t = days[i]

        mu_w, sigma_w = mus[w], sigmas[w]
        mu_b, sigma_b = mus[b], sigmas[b]

        dt_w = (t - last_played[w]) if last_played[w] >= 0.0 else -1.0
        dt_b = (t - last_played[b]) if last_played[b] >= 0.0 else -1.0

        phi_star_w = age_phi(phis[w], sigma_w, dt_w, rating_period_days, max_phi) if dt_w >= 0.0 else max_phi
        phi_star_b = age_phi(phis[b], sigma_b, dt_b, rating_period_days, max_phi) if dt_b >= 0.0 else max_phi

        phi_comb = math.sqrt(phi_star_w * phi_star_w + phi_star_b * phi_star_b)
        e_pred = expected_score(mu_w, mu_b, phi_comb)

        if i >= warmup_games:
            qualifies = True if min_idle_days <= 0.0 else ((dt_w >= min_idle_days) or (dt_b >= min_idle_days))
            if qualifies:
                p = min(max(e_pred, 1e-12), 1.0 - 1e-12)
                game_ll = -(s_w * math.log(p) + (1.0 - s_w) * math.log(1.0 - p))
                err = s_w - e_pred
                game_bs = err * err

                total_log_loss += game_ll
                total_brier += game_bs
                eval_count += 1

                player_ll_sum[w] += game_ll
                player_ll_sum[b] += game_ll
                player_bs_sum[w] += game_bs
                player_bs_sum[b] += game_bs
                player_eval_games[w] += 1
                player_eval_games[b] += 1

        new_mu_w, new_phi_w, new_sigma_w = update_player_m1(
            mu_w, phi_star_w, sigma_w, mu_b, phi_star_b, s_w, tau, min_phi, min_mu, max_mu
        )
        new_mu_b, new_phi_b, new_sigma_b = update_player_m1(
            mu_b, phi_star_b, sigma_b, mu_w, phi_star_w, 1.0 - s_w, tau, min_phi, min_mu, max_mu
        )

        mus[w], phis[w], sigmas[w], last_played[w] = new_mu_w, new_phi_w, new_sigma_w, t
        mus[b], phis[b], sigmas[b], last_played[b] = new_mu_b, new_phi_b, new_sigma_b, t

    macro_ll = 0.0
    macro_bs = 0.0
    eval_players = 0
    for u in range(n_players):
        c = player_eval_games[u]
        if c > 0:
            macro_ll += player_ll_sum[u] / c
            macro_bs += player_bs_sum[u] / c
            eval_players += 1

    game_ll_avg = (total_log_loss / eval_count) if eval_count > 0 else np.nan
    game_bs_avg = (total_brier / eval_count) if eval_count > 0 else np.nan
    macro_ll_avg = (macro_ll / eval_players) if eval_players > 0 else np.nan
    macro_bs_avg = (macro_bs / eval_players) if eval_players > 0 else np.nan

    return game_ll_avg, game_bs_avg, macro_ll_avg, macro_bs_avg, eval_count, eval_players

