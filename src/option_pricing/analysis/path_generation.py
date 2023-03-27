import numpy as np


def geometric_brownian_motion(
    S0,
    mu,
    sigma,
    T,
    number_steps,
    number_replications,
    vectorized=True,
):
    """This function generates paths of a financial asset basend on the geometric
    brwonian motion.

    Args:
        S0 (float): Starting value of the asset
        mu (float): Drift term of the asset
        sigma (float): Volatility of the asset
        T (int): Time horizon
        number_steps (int): Number of steps the function simulates
        number_replications (int): Number of different replications the function produces
        vectorized (bool, optional): If true, function uses vectorized approach. If false, function uses monte carlo approach. Defaults to True.

    Returns:
        array: Simulated path of the asset

    """
    # precompute constants
    dt = T / number_steps
    nu_dt = ((mu - pow(sigma, 2)) / 2) * dt
    sigma_sqrt_dt = sigma * np.sqrt(dt)

    if vectorized is True:

        GBM_errors = np.random.normal(
            nu_dt,
            sigma_sqrt_dt,
            (number_replications, number_steps),
        )
        GBM_start_errors = np.concatenate(
            (np.full((number_replications, 1), np.log(S0)), GBM_errors),
            axis=1,
        )
        GBM_path = np.cumsum(GBM_start_errors, axis=1)

        return np.exp(GBM_path)

    else:

        # preallocate array space
        GBM_path = np.empty((number_replications, number_steps + 1))
        GBM_path[:, 0] = S0

        # monte carlo simulation
        for replication in range(number_replications):
            for step in range(1, number_steps + 1):
                GBM_path[replication, step] = GBM_path[replication, step - 1] * np.exp(
                    np.random.normal(nu_dt, sigma_sqrt_dt),
                )

        return GBM_path

