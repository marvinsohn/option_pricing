# Variance Reduction Techniques of Monte Carlo Simulations for Option Pricing

[![image](https://img.shields.io/github/actions/workflow/status/marvinsohn/option_pricing/main.yml?branch=main)](https://github.com/marvinsohn/option_pricing/actions?query=branch%3Amain)
[![image](https://codecov.io/gh/marvinsohn/option_pricing/branch/main/graph/badge.svg)](https://codecov.io/gh/marvinsohn/option_pricing)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/marvinsohn/option_pricing/main.svg)](https://results.pre-commit.ci/latest/github/marvinsohn/option_pricing/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project implements variance reduction techniques of monte carlo simulations for
pricing European vanilla call options.

## Usage

To get started, make sure that the following programs are installed on your machine and
can be found on your path:

- Anaconda or Miniconda
- a LaTex distribution
- Git
- Text editor

If your machine is set up, open the terminal, navigate to the parent folder of where you
want to store this project, and execute the following code line by line:

```console
$ git clone https://github.com/marvinsohn/option_pricing
$ cd option_pricing
§ conda env create -f environment.yml
$ conda activate option_pricing
```

Now everything is set up! To build the project, type

````console
$ pytask
```

To run the test, type

```console
$ pytest
```

## Structure

After pytask is executed, the paper can be found under paper/option_pricing.pdf.
The python scripts relevant for the execution of the project can be found under src/option_pricing/.
The tests are stored under tests/.

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
````
