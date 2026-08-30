"""The registry's compiler: TOML records in, sympy and its printings out.

One registry entry is parsed once, and every downstream artifact is a
*printing* of that same object -- LaTeX for the docs, a JAX callable for the
oracle, an ``srepr`` hash for freshness. Nothing downstream is authored, so
nothing downstream can disagree.

Split from ``records``/``loader`` for organisation only. ``latex`` is an output
every record emits, so sympy is a core dependency and this module is always
available.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from .functions import ALLOWED_FUNCTIONS, JAX_FUNCTION_BINDINGS
from .records import (
    EquationRecord,
    RegistryError,
    ScalarBinding,
    SourceBundle,
    SymbolRecord,
)

_SUPPORTED_ASSUMPTIONS = frozenset(
    {"positive", "negative", "nonnegative", "real", "integer", "nonzero"}
)

#: The construction primitives sympy's own transformations emit into the code
#: they hand to ``eval``: ``auto_number`` writes ``Integer(4000)`` and
#: ``Float('4.65')``, ``repeated_decimals`` writes ``Rational``, ``auto_symbol``
#: writes ``Symbol``/``Function``. They must be reachable or the parse dies on
#: its own output -- but supplying *only* these keeps ``global_dict`` from
#: being the wide door onto sympy that its default (``from sympy import *``)
#: would be.
#:
#: ``rationalize`` is deliberately absent from the transformation set below:
#: it is what turns ``Float('4.65')`` into ``Rational(93, 20)``, which renders
#: as an unreadable fraction on a documentation page.
_CONSTRUCTORS: dict[str, Any] = {
    "Symbol": sympy.Symbol,
    "Function": sympy.Function,
    "Integer": sympy.Integer,
    "Float": sympy.Float,
    "Rational": sympy.Rational,
    "I": sympy.I,
}


def build_symbol(record: SymbolRecord) -> sympy.Symbol:
    """A sympy symbol carrying its declared assumptions.

    Assumptions are live, not decorative: with ``positive=True``,
    ``sqrt(L**2)`` simplifies to ``L``. They also survive ``srepr``, so two
    symbols named ``L`` with different assumptions hash differently -- which is
    what makes the freshness key trustworthy.
    """
    unknown = set(record.assumptions) - _SUPPORTED_ASSUMPTIONS
    if unknown:
        raise RegistryError(
            f"symbol {record.id}: unsupported assumptions {sorted(unknown)}"
        )
    return sympy.Symbol(record.name, **{name: True for name in record.assumptions})


def parse_expression(text: str, symbols: dict[str, sympy.Symbol]) -> sympy.Expr:
    """Parse a registry expression string into sympy.

    The name table is closed: only the equation's declared symbols and the
    functions in :data:`~startrax.registry.functions.ALLOWED_FUNCTIONS` are
    visible. A registry file ships inside the package, and ``parse_expr`` is
    otherwise a wide door -- an undeclared name must be a load error, not a
    freshly invented symbol.
    """
    local_dict: dict[str, Any] = {**ALLOWED_FUNCTIONS, **symbols}
    try:
        expression = parse_expr(
            text,
            local_dict=local_dict,
            global_dict=dict(_CONSTRUCTORS),
            transformations=standard_transformations,
            evaluate=True,
        )
    except (SyntaxError, TypeError, AttributeError) as error:
        raise RegistryError(f"cannot parse expression {text!r}: {error}") from error

    undeclared = sorted(
        symbol.name for symbol in expression.free_symbols if symbol.name not in symbols
    )
    if undeclared:
        raise RegistryError(
            f"expression uses undeclared symbols {undeclared}; declare them in "
            "symbol_ids or coefficient_ids"
        )
    return expression


@dataclass(frozen=True)
class BuiltEquation:
    """One registry equation, parsed once.

    ``symbols`` holds every free symbol keyed by its code-safe name -- the
    physical quantities, the fitted coefficients and the equation-bound
    constants alike, because sympy does not distinguish them. The registry
    does, and the distinction is the point:

    * ``variable_names`` -- what the caller supplies;
    * ``coefficient_names`` -- the paper's fitted numbers, each a citable
      record carrying its own sigma;
    * ``bound_names`` -- constants the *equation* binds, ``Zsun`` above all.
      Never reached for globally: every wind paper scales as ``Z/Zsun`` and
      each means a different ``Zsun``.

    So the expression stays the paper's formula while every number in it stays
    separately attributable.
    """

    record: EquationRecord
    expression: sympy.Expr
    symbols: dict[str, sympy.Symbol]
    variable_names: tuple[str, ...]
    coefficient_names: tuple[str, ...]
    coefficient_values: dict[str, float]
    bound_names: tuple[str, ...]
    bound_values: dict[str, float]
    latex_names: dict[sympy.Symbol, str]

    @property
    def srepr_sha256(self) -> str:
        return srepr_sha256(self.expression)

    @property
    def parameter_values(self) -> dict[str, float]:
        """Everything the kernel needs beyond its variables.

        Coefficients and bound constants together, so a caller supplies only
        the physical inputs and the registry supplies every number it owns.
        """
        return {**self.coefficient_values, **self.bound_values}


def build_equation(bundle: SourceBundle, equation_id: str) -> BuiltEquation:
    """Parse one equation from a loaded source bundle."""
    if equation_id not in bundle.equations:
        raise RegistryError(
            f"{bundle.source.id}: no equation {equation_id!r} "
            f"(have {sorted(bundle.equations)})"
        )
    record = bundle.equations[equation_id]
    if record.representation != "symbolic" or record.expression is None:
        raise RegistryError(
            f"{equation_id}: representation={record.representation!r} has no "
            "expression to build -- it carries provenance, not code"
        )

    symbol_records = bundle.symbols_for(record)
    coefficient_records = bundle.coefficients_for(record)

    collision = set(symbol_records) & set(coefficient_records)
    if collision:
        raise RegistryError(
            f"{equation_id}: {sorted(collision)} declared as both symbol and coefficient"
        )

    symbols: dict[str, sympy.Symbol] = {
        name: build_symbol(item) for name, item in symbol_records.items()
    }
    latex_names: dict[sympy.Symbol, str] = {
        symbols[name]: item.latex for name, item in symbol_records.items()
    }
    for name, coefficient in coefficient_records.items():
        symbols[name] = sympy.Symbol(name)
        if coefficient.latex:
            latex_names[symbols[name]] = coefficient.latex

    # Values bound by the equation itself -- Zsun above all. Never reached for
    # globally: every paper scales as Z/Zsun and each means a different Zsun.
    bound_values: dict[str, float] = {}
    for name, binding in record.symbol_bindings.items():
        if not isinstance(binding, ScalarBinding):
            continue  # a driver-only binding declares WHICH quantity, not a number
        if name in coefficient_records:
            raise RegistryError(
                f"{equation_id}: {name!r} is bound to a value and also declared as "
                "a coefficient -- a fitted number and a convention are different things"
            )
        bound_values[name] = binding.value
        symbols.setdefault(name, sympy.Symbol(name))

    expression = parse_expression(record.expression, symbols)

    used = {symbol.name for symbol in expression.free_symbols}
    unused = sorted((set(symbols) - used) & set(coefficient_records))
    if unused:
        raise RegistryError(
            f"{equation_id}: coefficients {unused} are declared but never used -- "
            "a coefficient nothing reads is exactly the inert-declaration pattern "
            "this registry exists to end"
        )

    return BuiltEquation(
        record=record,
        expression=expression,
        symbols=symbols,
        # A bound symbol is supplied by the registry, never by the caller, so
        # it leaves the variable list even though it is a declared symbol.
        variable_names=tuple(sorted((set(symbol_records) & used) - set(bound_values))),
        coefficient_names=tuple(sorted(set(coefficient_records) & used)),
        coefficient_values={
            name: item.value for name, item in coefficient_records.items()
        },
        bound_names=tuple(sorted(set(bound_values) & used)),
        bound_values={
            name: value for name, value in bound_values.items() if name in used
        },
        latex_names=latex_names,
    )


# ---------------------------------------------------------------------------
# printings
# ---------------------------------------------------------------------------


def srepr_sha256(expression: sympy.Expr) -> str:
    """The freshness key: a hash of the expression's exact internal structure.

    Not a hash of the registry text. ``c1*log10(L)+c3``, ``c1 * log10(L) + c3``
    and ``c3 + c1*log10(L)`` are three source strings, three text hashes and
    **one** ``srepr``. Hashing the text would mark a generated kernel stale for
    adding a space -- a forced regeneration with no physical change, and a
    staleness marker nobody would trust.
    """
    return hashlib.sha256(sympy.srepr(expression).encode("utf-8")).hexdigest()


def to_latex(built: BuiltEquation, **options: Any) -> str:
    """Render the equation for a documentation page.

    Display names come from the symbol records, so the code path can keep
    ``teff`` (a valid Python identifier, which ``lambdify`` will not dummify)
    while the page shows :math:`T_{\\rm eff}`.
    """
    return sympy.latex(
        built.expression,
        symbol_names=built.latex_names,
        **options,
    )


def partial_derivative(built: BuiltEquation, name: str) -> sympy.Expr:
    """The analytic partial with respect to one symbol.

    Measured on Antoniadis Eq. 4: this matches ``jax.grad`` at relative
    difference ``0.0``. The hand-derived ``value_and_partials`` it would
    replace has no detector for silently stopping to match its own primal.
    """
    if name not in built.symbols:
        raise RegistryError(
            f"{built.record.id}: no symbol {name!r} to differentiate by"
        )
    return sympy.diff(built.expression, built.symbols[name])


def lambdify_jax(
    built: BuiltEquation, expression: sympy.Expr | None = None
) -> Callable[..., Any]:
    """A JAX callable for the equation, or for a derivative of it.

    Arguments are the equation's variables followed by its coefficients, both
    in sorted order, and every name is a valid identifier so keyword calls
    work. ``log10`` is bound to ``jnp.log10`` rather than falling back to
    ``log(x)/log(10)``.

    Passing ``expression`` compiles some other printing of the same record --
    a :func:`partial_derivative`, above all -- against the *same* argument
    list, so a primal and its derivative are called identically.

    This is the *oracle* path. Generated production kernels are committed
    source, never produced at import: ``lambdify`` uses ``exec``, and
    generating at runtime would mean executing a registry string on every
    import of the package.
    """
    import jax.numpy as jnp

    bindings = {
        name: getattr(jnp, attribute)
        for name, attribute in JAX_FUNCTION_BINDINGS.items()
        if hasattr(jnp, attribute)
    }
    arguments = [
        built.symbols[name]
        for name in (
            *built.variable_names,
            *built.coefficient_names,
            *built.bound_names,
        )
    ]
    target = built.expression if expression is None else expression
    return sympy.lambdify(arguments, target, modules=[bindings, "jax"])


def resolve_oracle_inputs(
    bundle: SourceBundle,
    equation_id: str,
    inputs: dict[str, Any],
    *,
    _stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Add registered sub-relation outputs to an oracle's external inputs."""
    if equation_id in _stack:
        cycle = " -> ".join((*_stack, equation_id))
        raise RegistryError(f"derived-equation cycle: {cycle}")
    built = build_equation(bundle, equation_id)
    values = dict(inputs)
    for symbol, derived in built.record.derived_symbols.items():
        if symbol in values:
            raise RegistryError(
                f"{equation_id}: caller supplied derived symbol {symbol!r}; "
                "its registered producer must own the value"
            )
        values[symbol] = evaluate_oracle(
            bundle, derived, values, _stack=(*_stack, equation_id)
        )
    missing = sorted(name for name in built.variable_names if name not in values)
    if missing:
        raise RegistryError(f"{equation_id}: missing oracle inputs {missing}")
    return values


def evaluate_oracle(
    bundle: SourceBundle,
    equation_id: str,
    inputs: dict[str, Any],
    *,
    _stack: tuple[str, ...] = (),
) -> Any:
    """Evaluate a registry oracle, recursively resolving declared sub-relations.

    This is deliberately an oracle-only convenience.  It does not put SymPy or
    runtime registry traversal into a production wind kernel; it makes the
    dependency graph itself testable instead of duplicating it in test code.
    """
    built = build_equation(bundle, equation_id)
    values = resolve_oracle_inputs(bundle, equation_id, inputs, _stack=_stack)
    kernel = lambdify_jax(built)
    return kernel(
        *(values[name] for name in built.variable_names),
        *(built.coefficient_values[name] for name in built.coefficient_names),
        *(built.bound_values[name] for name in built.bound_names),
    )


def verify_identity(bundle: SourceBundle, equation_id: str) -> tuple[sympy.Expr, ...]:
    """Return the SIMPLIFIED residuals of a registered derivation. All zero means true.

    A **different check from an oracle**, and neither subsumes the other. An oracle
    proves an implementation matches the expression the registry holds; an identity
    proves the expression is *algebraically true* -- that the stated relation follows
    from the ones it was derived from. An implementation can faithfully realise a wrong
    derivation, and a correct derivation can be implemented wrongly.

    ``identity_given`` states the hypothesis a derivation holds under, as substitutions
    applied before simplifying. The mass-coordinate slope identity is true only *given*
    mass continuity ``Derivative(m(r), r) = 4 pi r^2 rho``; recording that as data is
    strictly more informative than the ``.subs()`` call inside a Python residual
    function, where it was invisible to everything except a reader of the source.

    ``doit()`` runs before simplification so unevaluated ``Derivative`` and ``limit``
    nodes -- which is how a derivation is naturally *written* -- actually evaluate.
    """
    record = bundle.equations.get(equation_id)
    if record is None:
        raise RegistryError(f"no equation {equation_id!r} in bundle")
    if not record.identity:
        raise RegistryError(f"{equation_id}: no identity to verify")

    symbol_records = bundle.symbols_for(record)
    coefficient_records = bundle.coefficients_for(record)
    symbols: dict[str, sympy.Symbol] = {
        name: build_symbol(item) for name, item in symbol_records.items()
    }
    for name in coefficient_records:
        symbols.setdefault(name, sympy.Symbol(name))

    given = [
        (parse_expression(target, symbols), parse_expression(replacement, symbols))
        for target, replacement in record.identity_given.items()
    ]

    residuals = []
    for text in record.identity:
        residual = parse_expression(text, symbols)
        # doit() BEFORE substituting, and again after. A hypothesis is usually
        # about a derivative that does not exist as a node until the surrounding
        # expression is evaluated: `Derivative(log(m(r)/r), r)` contains no
        # `Derivative(m(r), r)` to match until it expands. Substituting only
        # beforehand would silently no-op, and the identity would then read as
        # FALSE rather than as unapplied -- a failure mode that looks like a
        # broken derivation instead of a broken hypothesis.
        residual = residual.doit()
        for target, replacement in given:
            residual = residual.subs(target, replacement)
        residuals.append(sympy.simplify(residual.doit()))
    return tuple(residuals)


def identity_holds(bundle: SourceBundle, equation_id: str) -> bool:
    """``True`` when EVERY registered residual simplifies to exactly zero."""
    return all(residual == 0 for residual in verify_identity(bundle, equation_id))
