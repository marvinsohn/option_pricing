"""Tasks running the core analyses."""

import time

import numpy as np
import pandas as pd
import pytask

from option_pricing.analysis.control_variates import cv_european_vanilla
from option_pricing.config import BLD


@pytask.mark.depends_on(BLD / "random_option_parameters.pkl")
@pytask.mark.produces(BLD / "results_option_pricing.pkl")
def task_compute_option_prices(depends_on, produces):
    """Compute option prices with different control variates."""
    option_parameters = pd.read_pickle(depends_on)

    control_variates = ["none", "antithetic", "delta", "gamma", "all"]

    results_option_pricing = pd.DataFrame(
        columns=[
            "control_variate",
            "elapsed_time",
            "average_standard_error",
        ],
    )

    for control_variate in control_variates:

        start_function_time = time.time()

        standard_error = [
            (
                cv_european_vanilla(
                    s0=s0,
                    k=k,
                    sigma=sigma,
                    r=r,
                    t=t,
                    number_steps=number_steps,
                    number_replications=number_replications,
                    option_type="call",
                    variate=control_variate,
                )
            )
            for (s0, k, sigma, r, t, number_steps, number_replications) in zip(
                option_parameters["s0"],
                option_parameters["k"],
                option_parameters["sigma"],
                option_parameters["r"],
                option_parameters["t"],
                option_parameters["number_steps"],
                option_parameters["number_replications"],
                strict=True,
            )
        ]

        elapsed_function_time = time.time() - start_function_time

        new_results_option_pricing = pd.Series(
            {
                "control_variate": control_variate,
                "elapsed_time": elapsed_function_time,
                "average_standard_error": np.average(standard_error),
            },
        )

        results_option_pricing = pd.concat(
            [results_option_pricing, new_results_option_pricing.to_frame().T],
            ignore_index=True,
        )

    results_option_pricing.to_pickle(produces)
