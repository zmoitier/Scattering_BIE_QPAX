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

  - [Matplotlib](https://github.com/matplotlib/matplotlib),
  - [NumPy](https://github.com/numpy/numpy),
  - [SciPy](https://github.com/scipy/scipy),
  - [seaborn](https://github.com/mwaskom/seaborn).

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
