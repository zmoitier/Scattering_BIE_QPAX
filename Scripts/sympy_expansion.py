from os import get_terminal_size

import sympy as sy

order = 3  # order of the expansion
nb_ter = get_terminal_size()[0]

ε, δ, h = sy.symbols("ε δ h", real=True, positive=True)
s, t, X = sy.symbols("s t X", real=True)
μ_term = sy.symbols(f"μ0:{order}", cls=sy.Function)


def box_print(string):
    print("-" * nb_ter)
    print(f"| {string} |")
    line = "-" * (len(string) + 2)
    print(f"+{line}+")


def inner_expansion():
    εKL = -(ε ** 2) / (2 * sy.pi * (1 + ε ** 2 + (1 - ε ** 2) * sy.cos(s + t)))
    εKL_series = εKL.subs(t, sy.pi - s + ε * X).series(x=ε, x0=0, n=order).expand()

    μ_series = sum((ε * X) ** i * μ_(sy.pi - s) for i, μ_ in enumerate(μ_term)) + sy.O(
        ε ** order
    )

    expr = (εKL_series * μ_series).expand()

    box_print("Inner expansion")
    int_expr = sy.O(ε ** 2)
    for i in range(order):
        expr_term = expr.coeff(ε, i)
        for u_ in μ_term:
            expr_term_u = expr_term.coeff(u_(sy.pi - s), 1).factor()
            int_term = sy.integrate(
                expr_term_u, (X, -δ / (2 * ε), δ / (2 * ε))
            ).expand()
            int_expr += (int_term * u_(sy.pi - s) * ε ** i).factor().expand()

    sy.pprint(int_expr)


inner_expansion()
