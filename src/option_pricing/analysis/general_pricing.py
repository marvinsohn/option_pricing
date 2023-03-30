"""General option pricing functions.

This file refers for the different specific implementations of the option pricing
formulas to the imported functions from other files in the option_pricing/analysis
directory.

"""
from option_pricing.analysis.black_scholes import bs_european_vanilla
from option_pricing.analysis.monte_carlo import mc_european_vanilla


def get_price_european_vanilla(
    s0,
    k,
    t,
    r,
    sigma,
    option_type,
    method,
    number_steps,
    number_replications,
):
    """Get the price of an European option.

    Args:
        s0 (float): Start value
        k (float): Strike price
        t (float): Time until expiry
        r (float): Risk-free rate
        sigma (float): volatility
        option_type (string): call, put
        method (string): black_scholes, monte_carlo_plain, monte_carlo_vectorized
        number_steps (int): Number of steps for monte carlo simulation
        number_replications (int): Number of monte carlo simulations

    Returns:
        float: Price of the European option.

    """
    if method == "black_scholes":

        return bs_european_vanilla(
            s0=s0,
            k=k,
            t=t,
            r=r,
            sigma=sigma,
            option_type=option_type,
        )

    if method == "monte_carlo_plain":

        result = mc_european_vanilla(
            s0=s0,
            k=k,
            t=t,
            r=r,
            sigma=sigma,
            number_steps=number_steps,
            number_replications=number_replications,
            vectorized=False,
            option_type=option_type,
        )

        return result[0]

    if method == "monte_carlo_vectorized":

        result = mc_european_vanilla(
            s0=s0,
            k=k,
            t=t,
            r=r,
            sigma=sigma,
            number_steps=number_steps,
            number_replications=number_replications,
            vectorized=True,
            option_type=option_type,
        )

        return result[0]

    return None
