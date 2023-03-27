"""Price options based on the analytical formulas from Black-Scholes."""
import numpy as np
from scipy.stats import norm


def european_vanilla(s0, k, t, r, sigma, call):
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


def european_exchange(v0, u0, sigma_v, sigma_u, rho, t):
    """Compute the value of an European exchange option (= spread option with K=0).

    Args:
        v0 (_type_): Drift of asset V
        u0 (_type_): Dirft of asset U
        sigma_v (float): Volatility of asset v
        sigma_u (_type_): Volatility of asset u
        rho (float): Correlation of the two Wiener processes
        t (integer): Time until expiry

    Returns:
        float: Option prize

    """
    sigma_hat = np.srqt(pow(sigma_u, 2) + pow(sigma_v, 2) - 2 * rho * sigma_u * sigma_v)
    d1 = (np.log(v0 / u0) + 0.5 * t * pow(sigma_hat)) / (sigma_hat * np.sqrt(t))
    d2 = d1 - sigma_hat * np.sqrt(t)

    return v0 * norm.cdf(d1) - u0 * norm.cdf(d2)
