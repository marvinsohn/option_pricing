"""Monte carlo option pricing with the help of control variates."""

import numpy as np
from scipy import stats

from option_pricing.analysis.monte_carlo import mc_european_vanilla


def cv_european_vanilla(
    s0,
    k,
    sigma,
    r,
    t,
    number_steps,
    number_replications,
    option_type,
    variate,
):
    """Price an European vanilla option with control variates.

    Args:
        s0 (float): Starting value underlying
        k (float): Strike price
        sigma (float): Volatility
        r (float): Risk-free rate
        t (float): Time until expiry
        number_steps (int): Number of steps for each simulation
        number_replications (int): Number of simulations
        option_type (string): call,put
        variate (string): none, antithetic, delta, gamma, all

    Returns:
        float: Estimated price of the option
        float: Standard error of the estimated option price

    """
    # compute constants
    dt = t / number_steps
    nu_dt = (r - 0.5 * pow(sigma, 2)) * dt
    sigma_sqrt_dt = sigma * np.sqrt(dt)
    ln_s0 = np.log(s0)

    # check for selected variate and return results of the appropriate pricing function
    if variate == "none":

        price, standard_error = mc_european_vanilla(
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

    elif variate == "antithetic":

        price, standard_error = cv_european_vanilla_antithetic(
            ln_s0=ln_s0,
            r=r,
            t=t,
            k=k,
            nu_dt=nu_dt,
            sigma_sqrt_dt=sigma_sqrt_dt,
            number_steps=number_steps,
            number_replications=number_replications,
            option_type=option_type,
        )

    elif variate == "delta":

        price, standard_error = cv_european_vanilla_delta()

    return price, standard_error


def cv_european_vanilla_antithetic(
    nu_dt,
    sigma_sqrt_dt,
    ln_s0,
    k,
    r,
    t,
    number_steps,
    number_replications,
    option_type,
):
    """Price an European vanilla option with antithetic control variate.

    Args:
        nu_dt (_type_): _description_
        sigma_sqrt_dt (_type_): _description_
        ln_s0 (_type_): _description_
        k (float): Strike price
        r (float): Risk-free rate
        t (float): Time until expiry
        number_steps (int): Number of steps in each simulation
        number_replications (int): Number of simulations
        option_type (string): call,put

    Returns:
        float: Estimated option price
        float: Standard error of the estimation

    """
    # Antithetic variate: simulate an asset, that is perfect negatively correlated
    # to the underlying of the option contract.

    # Monte carlo: simulate both assets,
    # the real underlying and the perfect negatively correlated one
    error_terms = np.random.normal(size=(number_steps, number_replications))
    delta_underlying = nu_dt + sigma_sqrt_dt * error_terms
    delta_negatively_correlated = nu_dt - sigma_sqrt_dt * error_terms

    # Compute the value at time t for both assets
    st_underlying = np.exp(ln_s0 + np.cumsum(delta_underlying, axis=0))
    st_negatively_correlated = np.exp(
        ln_s0 + np.cumsum(delta_negatively_correlated, axis=0),
    )

    # Compute option price at time t
    if option_type == "call":
        price_at_t = (
            1
            / 2
            * (
                np.maximum(0, st_underlying[-1] - k)
                + np.maximum(0, st_negatively_correlated[-1] - k)
            )
        )

    else:
        price_at_t = (
            1
            / 2
            * (
                np.maximum(0, k - st_underlying[-1])
                + np.maximum(0, k - st_negatively_correlated[-1])
            )
        )

    # Compute option price at time 0 and standard error
    price_at_0 = np.exp(-r * t) * np.sum(price_at_t) / number_replications
    sigma_option_price = np.sqrt(
        np.sum((np.exp(-r * t) * price_at_t - price_at_0) ** 2)
        / (number_replications - 1),
    )
    standard_error = sigma_option_price / np.sqrt(number_replications)

    return price_at_0, standard_error


def cv_european_vanilla_delta(
    s0,
    k,
    r,
    t,
    sigma,
    option_type,
    number_steps,
    number_replications,
    nu_dt,
    sigma_sqrt_dt,
    dt,
):
    """Price an European vanilla option with delta control variate.

    Args:
        s0 (float): Start value of underlying
        k (float): Strike price
        r (float): Risk-free rate
        t (float): Time until expiry
        sigma (float): volatility
        option_type (string): call, put
        number_steps (int): Number of steps in each simulation
        number_replications (int): Number of simulations
        nu_dt (_type_): _description_
        sigma_sqrt_dt (_type_): _description_
        dt (_type_): _description_

    Returns:
        float: Estimated price of the option
        float: Standard error of the estimate

    """
    # Monte carlo simulation
    error_terms = np.random.normal(size=(number_steps, number_replications))
    delta_underlying = nu_dt + sigma_sqrt_dt * error_terms
    st_underlying = s0 * np.cumprod(np.exp(delta_underlying), axis=0)
    st_underlying = np.concatenate(
        (np.full(shape=(1, number_replications), fill_value=s0), st_underlying),
    )
    delta_underlying = get_delta(
        r,
        st_underlying[:-1].T,
        k,
        np.linspace(t, dt, number_steps),
        sigma,
        option_type,
    ).T
    aux = np.cumsum(
        delta_underlying * (st_underlying[1:] - st_underlying[:-1] * np.exp(r * dt)),
        axis=0,
    )

    if option_type == "call":
        price_at_t = np.maximum(0, st_underlying[-1] - k) - aux[-1]
    else:
        price_at_t = np.maximum(0, k - st_underlying[-1]) - aux[-1]

    # Compute option price at time 0 and standard error
    price_at_0 = np.exp(-r * t) * np.sum(price_at_t) / number_replications
    sigma_option_price = np.sqrt(
        np.sum((np.exp(-r * t) * price_at_t - price_at_0) ** 2)
        / (number_replications - 1),
    )
    standard_error = sigma_option_price / np.sqrt(number_replications)

    return price_at_0, standard_error


def get_delta(s0, k, r, t, sigma, option_type):
    """Compute the delta of an option.

    Args:
        s0 (float): Start value of underlying
        k (float): Strike price
        r (float): Risk-free rate
        t (float): Time until expiry
        sigma (float): Volatility
        option_type (string): call, put

    Returns:
        float: delta of the option

    """
    d1 = (np.log(s0 / k) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))

    if option_type == "call":
        return stats.norm.cdf(d1, 0, 1)

    return -stats.norm.cdf(-d1, 0, 1)
