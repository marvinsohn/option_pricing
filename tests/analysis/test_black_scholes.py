"""Tests for the black-scholes pricing functions."""

import numpy as np
from option_pricing.analysis.black_scholes import (
    bs_european_vanilla,
)

desired_precision = 10e-2


def test_bs_european_vanilla():
    real_call_price = 4.08
    real_put_price = 1.16

    computed_call_price = bs_european_vanilla(
        s0=60,
        k=60,
        t=1,
        r=0.05,
        sigma=0.1,
        call=True,
    )
    computed_put_price = bs_european_vanilla(
        s0=60,
        k=60,
        t=1,
        r=0.05,
        sigma=0.1,
        call=False,
    )

    assert np.absolute([real_call_price - computed_call_price]) < desired_precision
    assert np.absolute([real_put_price - computed_put_price]) < desired_precision
