"""Generate paths of stochastic processes."""
import numpy as np
from scipy.linalg import cholesky


def get_geometric_brownian_motion(
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


def get_multivariate_geometric_brownian_motion(
    s0,
    mu,
    sigma,
    correlation,
    t,
    number_steps,
    number_replications,
):
    """Generate multivariate paths of geometric brownian motion.

    Args:
        s0 (float): List of starting values
        mu (float): List of drift values
        sigma (float): List of volatilities
        correlation (array): List of correlation matrices
        t (float): Time horizon
        number_steps (integer): Number of steps of each simulation
        number_replications (integer): Number of simulations

    Returns:
        array: simulated multivariate geometric brownian motions

    """
    # precompute constants
    dt = t / number_steps
    sigma_2 = [pow(individual_sigma, 2) for individual_sigma in sigma]
    divided_sigma = [individual_sigma_2 / 2 for individual_sigma_2 in sigma_2]
    subtracted = [
        individual_mu - individual_divided_sigma
        for (individual_mu, individual_divided_sigma) in zip(
            (mu, divided_sigma),
            strict=True,
        )
    ]
    nu_dt = [individual_subtracted * dt for individual_subtracted in subtracted]
    sigma_sqrt_dt = [np.sqrt(dt) * individual_sigma for individual_sigma in sigma]

    # Create Cholensky factor
    lower_cholensky = cholesky(correlation, lower=True)

    # 3 Dimensions: Every mu has a number of replications with a number of steps
    # Add +1 to number_steps, since the first steps already exixts with s0
    multivariate_gbm_path = np.empty((number_replications, number_steps + 1, len(mu)))
    # loop over every dimension
    for drift in range(len(mu)):
        # assign starting value of the paths
        multivariate_gbm_path[:, 0, drift] = s0[drift]
        for replication in range(number_replications):
            for step in range(1, number_steps + 1):
                # produce correlated Wiener process error terms
                e = np.matmul(np.random.normal(0, 1, size=len(mu)), lower_cholensky)

                multivariate_gbm_path[replication, step, :] = multivariate_gbm_path[
                    replication,
                    (step - 1),
                    :,
                ] * np.exp(nu_dt + sigma_sqrt_dt * e)

    return multivariate_gbm_path


def get_twodimensional_geometric_brownian_motion(s0, mu, sigma, rho, t, number_steps):
    """Generate twodimensional path of geometric brownian motion.

    Args:
        s0 (float): Start value of the underlyings
        mu (float): Drift
        sigma (float): Volatility
        rho (float): Correlation
        t (float): Time until expiry
        number_steps (integer): Number of steps in each simulation

    Returns:
        array: twodimensional geometric brownian motion path

    """
    # compute constants
    dt = t / number_steps
    [individual_sigma ^ 2 / 2 for individual_sigma in sigma]
    nu = [
        individual_mu - individual_sigma
        for individual_mu, individual_sigma in zip((mu, sigma), strict=True)
    ]
    nu_dt = [individual_nu * dt for individual_nu in nu]
    sigma_dt = [individual_sigma * dt for individual_sigma in sigma]
    sqrt_rho = np.sqrt(1 - pow(rho, 2))

    # pre-allocate array
    # array has number_steps + 1 columns, since first column is filled by s0
    twodimensional_gbm_path = np.empty((2, number_steps + 1))
    twodimensional_gbm_path[:, 0] = s0

    for step in range(1, number_steps + 1):

        # first GBM path
        e_0 = np.random.normal(size=1)
        twodimensional_gbm_path[0, step] = twodimensional_gbm_path[
            0,
            step - 1,
        ] * np.exp(nu_dt[0] + sigma_dt[0] * e_0)

        # second GBM path
        # this error term has to be correlated with the error term of the first GBM
        e_1 = np.random.normal(size=1) * sqrt_rho + rho * e_0
        twodimensional_gbm_path[1, step] = twodimensional_gbm_path[
            1,
            step - 1,
        ] * np.exp(nu_dt[1] + sigma_dt[1] * e_1)

    return twodimensional_gbm_path
