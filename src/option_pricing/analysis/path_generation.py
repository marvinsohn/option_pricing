"""Generate paths of stochastic processes."""
import numpy as np


def geometric_brownian_motion(
    s0,
    mu,
    sigma,
    t,
    number_steps,
    number_replications,
    vectorized,
):
    """Generate paths of the geometric brownian motion.

    Args:
        s0 (float): Starting value of the asset
        mu (float): Drift term of the asset
        sigma (float): Volatility of the asset
        t (int): Time horizon
        number_steps (int): Number of steps the function simulates
        number_replications (int): Number of replications the function produces
        vectorized (bool): If true, use vectorization. If false, use monte carlo.

    Returns:
        array: Simulated path of the asset

    """
    # precompute constants
    dt = t / number_steps
    nu_dt = ((mu - pow(sigma, 2)) / 2) * dt
    sigma_sqrt_dt = sigma * np.sqrt(dt)

    if vectorized is True:

        gbm_errors = np.random.normal(
            nu_dt,
            sigma_sqrt_dt,
            (number_replications, number_steps),
        )
        gbm_start_errors = np.concatenate(
            (np.full((number_replications, 1), np.log(s0)), gbm_errors),
            axis=1,
        )
        gbm_path = np.cumsum(gbm_start_errors, axis=1)

        return np.exp(gbm_path)

    # preallocate array space
    gbm_path = np.empty((number_replications, number_steps + 1))
    gbm_path[:, 0] = s0

    # monte carlo simulation
    for replication in range(number_replications):
        for step in range(1, number_steps + 1):
            gbm_path[replication, step] = gbm_path[replication, step - 1] * np.exp(
                np.random.normal(nu_dt, sigma_sqrt_dt),
            )

    return gbm_path


def multivariate_geoimetric_brownian_motion(
    s0,
    mu,
    sigma,
    correlation,
    t,
    number_steps,
    nummber_replications,
):
    """Generate multivariate paths of geometric brownian motion.

    Args:
        s0 (float): List of starting values
        mu (float): List of drift values
        sigma (float): List of volatilities
        correlation (array): List of correlation matrices
        t (float): Time horizon
        number_steps (integer): Number of steps of each simulation
        nummber_replications (_type_): Number of simulations

    Returns:
        array: simulated multivariate geometric brownian motions

    """
    # precompute constants
    dt = t / number_steps
    nu_dt = (mu - pow(sigma, 2) / 2) * dt
    sigma_sqrt_dt = np.sqrt(dt) * sigma

    # Create Cholensky factor
    lower_cholensky_factor = np.transpose(np.linalg.cholesky(correlation))

    # 3 Dimensions: Every mu has a number of replications with a number of steps
    # Add +1 to number_steps, since the first steps already exixts with s0
    multivariate_gbm_path = np.empty((nummber_replications, number_steps + 1, len(mu)))
    # loop over every dimension
    for drift in range(len(mu)):
        multivariate_gbm_path[:, 0, drift] = s0[drift]
        for replication in range(nummber_replications):
            for step in range(1, number_steps + 1):
                # produce correlated standarnd normal error terms
                e = np.matmul(lower_cholensky_factor, np.random.normal(size=len(mu)))
                multivariate_gbm_path[replication, step, :] = multivariate_gbm_path[
                    replication,
                    step - 1,
                    :,
                ] * np.exp(nu_dt + sigma_sqrt_dt * e)

    return multivariate_gbm_path
