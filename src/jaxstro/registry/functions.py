"""Functions the registry adds to sympy's vocabulary.

``log10`` is the whole of Phase 1, and it earns a class rather than a helper.
Out of the box ``sympy.latex(log(L, 10))`` renders as
:math:`\\frac{\\log L}{\\log 10}` -- correct, and unreadable in a paper. The
same expression lambdifies to ``log(x)/log(10)``, which is also numerically
worse than a direct base-10 logarithm.

Declaring ``log10`` as a first-class :class:`sympy.Function` carrying its own
``_latex`` and ``fdiff`` fixes both defects at once: the printer emits
:math:`\\log_{10}` and the code path maps to ``jnp.log10``.

Deliberately **not** auto-evaluating. The registry stores the paper's formula
verbatim, so ``log10(10)`` stays symbolic rather than collapsing to ``1`` --
what is hashed is what the paper wrote.

Adding a function here is the extension point every other package will need
(``erf``, ``expm1``, tabulated fits). Nothing in this module knows about
startrax.
"""

from __future__ import annotations

import sympy


class log10(sympy.Function):  # noqa: N801 - the name IS the code-path identifier
    """Base-10 logarithm as a first-class registry function.

    The class name is load-bearing twice over: ``sympy.lambdify`` prints the
    call by ``__name__``, and :func:`startrax.registry.symbolic.lambdify_jax`
    binds that name to ``jnp.log10``.
    """

    nargs = 1
    is_real = True

    def fdiff(self, argindex: int = 1):
        """d/dx log10(x) = 1 / (x ln 10).

        Without this, ``diff`` cannot differentiate the expression at all, so
        the generated analytic partials -- the output that replaces
        hand-derived ``value_and_partials`` -- would not exist.
        """
        if argindex != 1:
            raise sympy.core.function.ArgumentIndexError(self, argindex)
        return 1 / (self.args[0] * sympy.log(10))

    def _latex(self, printer, *args) -> str:
        """Render as ``\\log_{10}(x)``.

        ``LatexPrinter.printmethod`` is ``"_latex"``, so defining this method
        is what the printer dispatches to.
        """
        return r"\log_{10}{\left(%s \right)}" % printer._print(self.args[0])

    def _eval_evalf(self, prec: int):
        """Numeric evaluation, for anchor checks against the paper's own values."""
        argument = self.args[0]._eval_evalf(prec)
        if argument is None:
            return None
        return sympy.log(argument, 10)._eval_evalf(prec)

    def _eval_rewrite(self, rule, args, **hints):
        if rule is sympy.log:
            return sympy.log(args[0]) / sympy.log(10)
        return None


#: Names a registry ``expression`` string may refer to beyond its declared
#: symbols and coefficients. Everything the parser is allowed to see is listed
#: here: a registry file ships in ``src/``, and ``parse_expr`` should not be a
#: door onto arbitrary sympy.
ALLOWED_FUNCTIONS: dict[str, object] = {
    "log10": log10,
    "log": sympy.log,
    "exp": sympy.exp,
    "sqrt": sympy.sqrt,
    "Abs": sympy.Abs,
    "Min": sympy.Min,
    "Max": sympy.Max,
    "Piecewise": sympy.Piecewise,
    "true": sympy.true,
    "True": sympy.true,
}

#: How each allowed function is spelled on the JAX code path.
JAX_FUNCTION_BINDINGS: dict[str, str] = {
    "log10": "log10",
    "log": "log",
    "exp": "exp",
    "sqrt": "sqrt",
    "Abs": "abs",
    "Min": "minimum",
    "Max": "maximum",
}
