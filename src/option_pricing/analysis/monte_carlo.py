"""Price options via plain monte carlo simulations.

This script is ONLY for pricing options via plain monte carlo. If you want monte carlo
with control variates, go to control_variestes.py.

"""
import numpy as np
from scipy.stats import ttest_1samp

from option_pricing.analysis.path_generation import (
    get_geometric_brownian_motion,
    get_multivariate_geometric_brownian_motion,
)


def mc_european_vanilla(
    s0,
    k,
    t,
    r,
    sigma,
    number_steps,
    number_replications,
    vectorized,
    option_type,
):
    """Price European vanilla option via monte carlo simulation.

    Args:
        s0 (float): Starting value of the underlying
        k (float): Strike price
        t (integer): Time until expiry
        r (float): Risk-free rate
        sigma (float): Volatility of the underlying
        number_steps (int): Number of steps in each simulation
        number_replications (inter): Number of different simulations
        vectorized (bool): True for vectorized version. False for loop version
        option_type (bool): call, put

    Returns:
        float: Present value of the option contract
        float: Standard error of the price of the option contract

    """
    # get paths
    gbm_paths = get_geometric_brownian_motion(
        s0=s0,
        mu=r,
        sigma=sigma,
        t=t,
        number_steps=number_steps,
        number_replications=number_replications,
        vectorized=vectorized,
    )

    # compute payoffs
    if option_type == "call":
        payoffs_at_t = [np.maximum(path[-1] - k, 0) for path in gbm_paths]
    else:
        payoffs_at_t = [np.maximum(k - path[-1], 0) for path in gbm_paths]
    square_payoffs_at_t = [pow(payoff_at_t, 2) for payoff_at_t in payoffs_at_t]

    # compute option metrics
    option_value_at_0 = np.exp(-r * t) * sum(payoffs_at_t) / number_replications
    option_sigma = np.sqrt(
        (sum(square_payoffs_at_t) - pow(sum(payoffs_at_t), 2) / number_replications)
        * np.exp(-2 * r * t)
        / (number_replications - 1),
    )
    option_standard_error = option_sigma / np.sqrt(number_replications)

    return option_value_at_0, option_standard_error


def mc_european_exchange(
    s0,
    mu,
    sigma,
    correlation,
    t,
    number_steps,
    number_replications,
):
    """Price European exchange option via monte carlo simulation.

    Args:
        s0 (float): List of starting values
        mu (float): List of drifts
        sigma (float): List of individual volatilities
        correlation (array): Correlation matrix
        t (float): Time until expiry
        number_steps (int): Number of simulated steps
        number_replications (int): Number of different replications

    Returns:
        float: Estimated option price
        float: Confidence interval of the estimated option price

    """
    gbm_paths = get_multivariate_geometric_brownian_motion(
        s0=s0,
        mu=mu,
        sigma=sigma,
        correlation=correlation,
        t=t,
        number_steps=number_steps,
        number_replications=number_replications,
    )

    payoffs = np.maximum(0, gbm_paths[:, 1, 0] - gbm_paths[:, 1, 1])

    discount_factors = np.exp([-individual_mu * t for individual_mu in mu])

    discounted_payoffs = [
        discount_factor * payoff
        for (discount_factor, payoff) in zip((discount_factors, payoffs), strict=True)
    ]

    aux = ttest_1samp(discounted_payoffs)
    value = aux[1]
    ci = aux.confidence_interval()

    return value, ci
