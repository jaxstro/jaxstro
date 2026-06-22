"""Small PyTree linear operators for matrix-free algebra."""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class LinearOperator(Protocol):
    """Protocol shared by the small operator classes in this module."""

    @property
    def shape(self) -> tuple[int, int]: ...

    def matvec(self, x: Float[Array, " n"]) -> Float[Array, " m"]: ...

    def rmatvec(self, x: Float[Array, " m"]) -> Float[Array, " n"]: ...

    def to_dense(self) -> Float[Array, " m n"]: ...


class DenseOperator(eqx.Module):
    """Linear operator backed by a dense matrix."""

    matrix: Float[Array, "m n"]

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def matvec(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.matrix @ x

    def rmatvec(self, x: Float[Array, " m"]) -> Float[Array, " n"]:
        return self.matrix.T @ x

    def to_dense(self) -> Float[Array, " m n"]:
        return self.matrix


class DiagonalOperator(eqx.Module):
    """Linear operator backed by a diagonal vector."""

    diagonal: Float[Array, "n"]

    @property
    def shape(self) -> tuple[int, int]:
        n = self.diagonal.shape[0]
        return (n, n)

    def matvec(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        return self.diagonal * x

    def rmatvec(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        return self.diagonal * x

    def to_dense(self) -> Float[Array, " n n"]:
        return jnp.diag(self.diagonal)


class ScaledOperator(eqx.Module):
    """Scalar multiple of an operator."""

    operator: LinearOperator
    scalar: Float[Array, ""]

    @property
    def shape(self) -> tuple[int, int]:
        return self.operator.shape

    def matvec(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.scalar * self.operator.matvec(x)

    def rmatvec(self, x: Float[Array, " m"]) -> Float[Array, " n"]:
        return self.scalar * self.operator.rmatvec(x)

    def to_dense(self) -> Float[Array, " m n"]:
        return self.scalar * self.operator.to_dense()


class SumOperator(eqx.Module):
    """Sum of two operators with identical shape."""

    left: LinearOperator
    right: LinearOperator

    @property
    def shape(self) -> tuple[int, int]:
        return self.left.shape

    def matvec(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.left.matvec(x) + self.right.matvec(x)

    def rmatvec(self, x: Float[Array, " m"]) -> Float[Array, " n"]:
        return self.left.rmatvec(x) + self.right.rmatvec(x)

    def to_dense(self) -> Float[Array, " m n"]:
        return self.left.to_dense() + self.right.to_dense()


class ProductOperator(eqx.Module):
    """Composition ``left @ right`` without materializing the product."""

    left: LinearOperator
    right: LinearOperator

    @property
    def shape(self) -> tuple[int, int]:
        return (self.left.shape[0], self.right.shape[1])

    def matvec(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.left.matvec(self.right.matvec(x))

    def rmatvec(self, x: Float[Array, " m"]) -> Float[Array, " n"]:
        return self.right.rmatvec(self.left.rmatvec(x))

    def to_dense(self) -> Float[Array, " m n"]:
        return self.left.to_dense() @ self.right.to_dense()


class TransposeOperator(eqx.Module):
    """Transpose view of an operator."""

    operator: LinearOperator

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.operator.shape
        return (cols, rows)

    def matvec(self, x: Float[Array, " m"]) -> Float[Array, " n"]:
        return self.operator.rmatvec(x)

    def rmatvec(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.operator.matvec(x)

    def to_dense(self) -> Float[Array, " n m"]:
        return self.operator.to_dense().T


class BlockDiagonalOperator(eqx.Module):
    """Block-diagonal composition of operators."""

    blocks: tuple[LinearOperator, ...]

    @property
    def shape(self) -> tuple[int, int]:
        rows = sum(block.shape[0] for block in self.blocks)
        cols = sum(block.shape[1] for block in self.blocks)
        return (rows, cols)

    def matvec(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        parts = []
        offset = 0
        for block in self.blocks:
            width = block.shape[1]
            parts.append(block.matvec(x[offset : offset + width]))
            offset += width
        return jnp.concatenate(parts, axis=0)

    def rmatvec(self, x: Float[Array, " m"]) -> Float[Array, " n"]:
        parts = []
        offset = 0
        for block in self.blocks:
            height = block.shape[0]
            parts.append(block.rmatvec(x[offset : offset + height]))
            offset += height
        return jnp.concatenate(parts, axis=0)

    def to_dense(self) -> Float[Array, " m n"]:
        dtype = self.blocks[0].to_dense().dtype
        dense = jnp.zeros(self.shape, dtype=dtype)
        row_offset = 0
        col_offset = 0
        for block in self.blocks:
            block_dense = block.to_dense()
            rows, cols = block.shape
            dense = dense.at[
                row_offset : row_offset + rows,
                col_offset : col_offset + cols,
            ].set(block_dense)
            row_offset += rows
            col_offset += cols
        return dense


def scale(operator: LinearOperator, scalar: float | Float[Array, ""]) -> ScaledOperator:
    """Return a scalar multiple of ``operator``."""
    return ScaledOperator(operator=operator, scalar=jnp.asarray(scalar))


def add(left: LinearOperator, right: LinearOperator) -> SumOperator:
    """Return the sum of two same-shape operators."""
    if left.shape != right.shape:
        msg = f"operator shapes must match for add: {left.shape} != {right.shape}"
        raise ValueError(msg)
    return SumOperator(left=left, right=right)


def compose(left: LinearOperator, right: LinearOperator) -> ProductOperator:
    """Return the composition ``left @ right``."""
    if left.shape[1] != right.shape[0]:
        msg = f"operator shapes do not compose: {left.shape} and {right.shape}"
        raise ValueError(msg)
    return ProductOperator(left=left, right=right)


def transpose(operator: LinearOperator) -> TransposeOperator:
    """Return a transpose view of ``operator``."""
    return TransposeOperator(operator=operator)


def block_diag(*blocks: LinearOperator) -> BlockDiagonalOperator:
    """Return a block-diagonal operator from one or more blocks."""
    if not blocks:
        msg = "block_diag requires at least one block"
        raise ValueError(msg)
    return BlockDiagonalOperator(blocks=tuple(blocks))


__all__ = [
    "LinearOperator",
    "DenseOperator",
    "DiagonalOperator",
    "ScaledOperator",
    "SumOperator",
    "ProductOperator",
    "TransposeOperator",
    "BlockDiagonalOperator",
    "scale",
    "add",
    "compose",
    "transpose",
    "block_diag",
]
