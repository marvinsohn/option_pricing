"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from option_pricing.config import BLD


@pytask.mark.depends_on(BLD / "results_option_pricing.pkl")
@pytask.mark.produces(BLD / "table_results.tex")
def task_compute_option_prices(depends_on, produces):
    """Store a table in LaTeX format with the comparison results."""
    results_option_pricing = pd.read_pickle(depends_on)

    # Rename columns to make them more readable
    results_option_pricing = results_option_pricing.rename(
        columns={
            "control_variate": "Variate",
            "absolute_computation_time": "Computation Time",
            "average_standard_error": "Standard Error",
            "computation_time_reduction_multiple": "Computation Time Multiple",
            "standard_error_reduction_multiple": "Standard Error Multiple",
        },
    )

    results_option_pricing = results_option_pricing.set_index("Variate")

    with open(produces, "w") as f:
        f.writelines(
            results_option_pricing.style.set_table_styles(
                [
                    {"selector": "toprule", "props": ":hline;"},
                    {"selector": "midrule", "props": ":hline;"},
                    {"selector": "bottomrule", "props": ":hline;"},
                ],
            ).to_latex(),
        )
