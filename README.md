[![DOI](https://zenodo.org/badge/357627837.svg)](https://zenodo.org/badge/latestdoi/357627837)

# Scattering_BIE_QPAX

Code for computing the scattering by a high aspect ratio ellipse using boundary integral equation methods.

## Reference

This is the code associated with the article:

- C. Carvalho, A. D. Kim, L. Lewis, and Z. Moitier, _Quadrature by Parity Asymptotic eXpansions (QPAX) for scattering by high aspect ratio particles_. [in preparation]

## Requirements

- Python version:

  - Tested on Python 3.8;
  - Should works on Python 3.7 but not tested.

- Require the following libraries:

  - Numerical computation: [NumPy](https://github.com/numpy/numpy) (≥ 1.20.2) and [SciPy](https://github.com/scipy/scipy) (≥ 1.6.2);

  - Visualization; [Matplotlib](https://github.com/matplotlib/matplotlib) (≥ 3.4.1) and [seaborn](https://github.com/mwaskom/seaborn) (≥ 0.11.1);

  - Formal computation: [SymPy](https://github.com/sympy/sympy) (≥ 1.8).

Might works with previous versions of the libraries but if it does not works try to update the library for example through pip

```bash
python3 -m pip install --user --upgrade matplotlib numpy seaborn scipy sympy
```

## Install

Clone from GitHub repository:

```bash
git clone https://github.com/zmoitier/Scattering_BIE_QPAX.git
```

## Instructions for usage

### Notebook version

Run the Jupyter notebook `run_fig_n.ipynb` to get the corresponding figure `n` in the paper.

### Script version

In the current directory run

```bash
python3 -m Scripts.run_fig_n
```

to get the corresponding figure `n` in the paper.
