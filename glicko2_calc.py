import math
from numba import njit

# Exact constants from lila (lichess)
GLICKO2_SCALE = 173.7178               # 400 / ln(10)
INIT_RATING = 1500.0
MIN_RATING = 400.0
MAX_RATING = 4000.0

MIN_RD = 45.0
MAX_RD = 500.0                         # lila: maxDeviation = 500d
PROVISIONAL_RD = 110.0                 # lila: ? mark threshold
STANDARD_RANKABLE_RD = 75.0            # lila: standardRankableDeviation = 75
VARIANT_RANKABLE_RD = 65.0             # lila: variantRankableDeviation = 65
STANDARD_COLOR_ADV_ELO = 11.782457     # lila: ColorAdvantage.standard = 11.782457d
INIT_VOL = 0.09                        # lila: defaultVolatility = 0.09d
MAX_VOL = 0.1                          # lila: maxVolatility = 0.1d
TAU = 0.75                             # lila system constant
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
def age_phi(phi: float, sigma: float, dt_days: float, rating_period_days: float, max_phi: float) -> float:
    if dt_days <= 0.0:
        return phi
    periods = dt_days / rating_period_days
    phi_star = math.sqrt(phi * phi + periods * sigma * sigma)
    return max_phi if phi_star > max_phi else phi_star


@njit(fastmath=True, cache=True)
def update_volatility(phi: float, sigma: float, v: float, delta: float, tau: float) -> float:
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
    mu: float, phi_star: float, sigma: float, mu_opp: float, phi_star_opp: float,
    score: float, tau: float, min_phi: float, min_mu: float, max_mu: float,
):
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

