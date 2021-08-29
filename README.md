[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4692601.svg)](https://doi.org/10.5281/zenodo.4692601)

# Scattering_BIE_QPAX

Quadrature by Parity eXpansions method for computing the scattering by a high aspect ratio ellipse using boundary integral equation methods.

## Reference

Codes are associated with the article:

- C. Carvalho, A. D. Kim, L. Lewis, and Z. Moitier, _Quadrature by Parity Asymptotic eXpansions (QPAX) for scattering by high aspect ratio particles_. [[arXiv](https://arxiv.org/abs/2105.02136)]

The bibtex entry can be access by the command.

```bash
python3 -c "from scr import __bibtex__; print(__bibtex__)"
@misc{carvalho2021quadrature,
      title={Quadrature by Parity Asymptotic eXpansions (QPAX) for scattering by high aspect ratio particles},
      author={Camille Carvalho and Arnold D. Kim and Lori Lewis and Zoïs Moitier},
      year={2021},
      eprint={2105.02136},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```

## Requirements

- Python version:

  - Tested on Python 3.9;
  - Should works on Python ≥3.7 but not tested.

- Require the following libraries:

  - For numerical computations: [NumPy](https://github.com/numpy/numpy) and [SciPy](https://github.com/scipy/scipy);

  - For symbolic computations: [SymPy](https://github.com/sympy/sympy);

  - For visualization: [Matplotlib](https://github.com/matplotlib/matplotlib) and [seaborn](https://github.com/mwaskom/seaborn);

  - For Jupyter notebook: [IPython](https://github.com/ipython/ipython) and [JupyterLab](https://github.com/jupyterlab/jupyterlab).

Might works with previous versions of the libraries but if it does not works try to update the libraries for example through pip

```bash
python3 -m pip install --user --upgrade -r requirements.txt
```

## Install

Clone from GitHub repository:

```bash
git clone https://github.com/zmoitier/Scattering_BIE_QPAX.git
```

## Instructions for usage

Run the Jupyter notebook `run_fig_n.ipynb` to get the corresponding figure `n` in the paper.

## Symbolic expressions

The Symbolic_expansions folder provides inner expansions for Laplace and Helmholtz double-layer potentials using Mathematica notebook or SymPy (see section 3 and 4 or the associated manuscript).

## Contact

If you have any questions or suggestions please feel free to create [an issue in this repository](https://github.com/zmoitier/Scattering_BIE_QPAX/issues/new).
