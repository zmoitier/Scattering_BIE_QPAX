""" Codes associated with the article:
    Quadrature by Parity Asymptotic eXpansions (QPAX) for scattering by high aspect
    ratio particles.
    C. Carvalho, A. D. Kim, L. Lewis, and Z. Moitier

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

__bibtex__ = """@misc{carvalho2021quadrature,
      title={Quadrature by Parity Asymptotic eXpansions (QPAX) for scattering by high aspect ratio particles}, 
      author={Camille Carvalho and Arnold D. Kim and Lori Lewis and Zoïs Moitier},
      year={2021},
      eprint={2105.02136},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}"""

from . import analytic, bie
from .boundary import Boundary
from .ellipse import Ellipse
from .mpl_pretty_plot import set_rcParams, set_size
from .trace import normal_trace, scaled_normal_trace, trace
from .trace_expansion import trace_dirichlet_expansion, trace_neumann_expansion
from .utils_plot import add_elem, logspace_epsilon, logspace_quadrature
