"""Task to produce random option parameters."""
import numpy as np
import pandas as pd
import pytask

from option_pricing.config import BLD


@pytask.mark.produces(BLD / "data.pkl")
def task_create_random_option_parameters(produces):
    """Produce one-dimensional random option parameters."""
    np.random.seed(42)

    s0 = 100 * np.random.random_sample(size=100)
    k = 150 * np.random.random_sample(size=100)
    t = np.random.random_sample(size=100)
    r = 0.1 * np.random.random_sample(size=100)
    sigma = 0.2 * np.random.random_sample(size=100)
    number_steps = np.random.random_integers(low=100, high=1000, size=100)
    number_replications = np.random.random_integers(low=50, high=500, size=100)

    option_parameters = pd.DataFrame(
        {
            "s0": s0,
            "k": k,
            "t": t,
            "r": r,
            "sigma": sigma,
            "option_type": ["call"] * len(s0),
            "number_steps": number_steps,
            "number_replications": number_replications,
        },
    )
    option_parameters.to_pickle(produces)
