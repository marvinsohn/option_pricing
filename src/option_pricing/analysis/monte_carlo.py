"""Price options via monte carlo simulations."""
import numpy as np
from scipy.stats import norm, ttest_1samp


def mc_european_vanilla(s0, k, t, r, sigma, number_replications, confidence_level):
    """Price European vanilla option via monte carlo simulation.

    Args:
        s0 (float): Starting value of the underlying
        k (float): Strike price
        t (integer): Time until expiry
        r (float): Risk-free rate
        sigma (float): Volatility of the underlying
        number_replications (inter): Number of different replications
        confidence_level (float): Confidence interval of the simulation

    Returns:
        _type_: _description_

    """
    nu_t = (r - 0.5 * pow(sigma, 2)) * t
    sigma_t = sigma * np.sqrt(t)
    st = s0 * np.exp(np.random.normal(nu_t, sigma_t, number_replications))
    d_payoff = np.exp(-r * t) * norm(np.maximum(st - k, 0))
    aux = ttest_1samp(d_payoff)
    value = aux[1]
    ci = aux.confidence_interval(confidence_level=confidence_level)

    return value, ci
