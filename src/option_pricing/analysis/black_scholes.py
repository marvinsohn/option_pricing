"""Price options based on the analytical formulas from Black-Scholes."""
import numpy as np
from scipy.stats import norm


def bs_european_vanilla(s0, k, t, r, sigma, call):
    """Compute the value of an European vanilla option.

    Args:
        s0 (float): Starting value of the underlying
        k (float): Strike price
        t (integer): Time until expiry
        r (float): Risk-free rate
        sigma (float): volatility
        call (bool, optional): If the option is a call option, select True.
                                If the option is a put option, select False.

    Returns:
        float: Option price

    """
    d1 = (np.log(s0 / k) + (r + pow(sigma, 2) / 2) * t) / (np.sqrt(t) * sigma)
    d2 = d1 - np.sqrt(t) * sigma

    if call is True:

        return s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    return k * np.exp(-r * t) * (1 - norm.cdf(d2)) - s0 * (1 - norm.cdf(d1))


def bs_european_exchange(v0, u0, sigma_v, sigma_u, rho, t):
    """Compute the value of an European exchange option.

    The option holder receives asset V in exchange for asset U.

    Args:
        v0 (_type_): Price of asset V
        u0 (_type_): Price of asset U
        sigma_v (float): Volatility of asset V
        sigma_u (_type_): Volatility of asset U
        rho (float): Correlation of the two Wiener processes
        t (float): Time until expiry

    Returns:
        float: Option prize

    """
    sigma_v_u = np.sqrt(pow(sigma_u, 2) + pow(sigma_v, 2) - 2 * rho * sigma_u * sigma_v)
    d1 = (np.log(v0 / u0) + 0.5 * t * pow(sigma_v_u, 2)) / (sigma_v_u * np.sqrt(t))
    d2 = d1 - sigma_v_u * np.sqrt(t)

    return v0 * norm.cdf(d1) - u0 * norm.cdf(d2)
