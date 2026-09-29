"""Simple structs for numpy arrays or gurobi variables with their bounds."""

from __future__ import annotations

import logging

import gurobipy as gp
import numpy as np

LOGGER = logging.getLogger(__name__)


class BoundedVar:
    """A wrapper for passing MVars / MLinExprs / MQuadExprs and bounds."""

    def __init__(
        self,
        *,
        n: np.ndarray,
        l: np.ndarray | float | int | None = None,
        u: np.ndarray | float | int | None = None,
        v: gp.MVar | gp.MLinExpr | gp.MQuadExpr | np.ndarray | None = None,
    ):
        """Initialise the BoundedVar."""
        if isinstance(l, int | float):
            l = l * np.ones(n.shape, dtype=n.dtype)
        if isinstance(u, int | float):
            u = u * np.ones(n.shape, dtype=n.dtype)

        self.n: np.ndarray = n
        self.l: np.ndarray = l if l is not None else n
        self.u: np.ndarray = u if u is not None else n
        self.v: gp.MVar | gp.MLinExpr | gp.MQuadExpr | np.ndarray = (
            v if v is not None else n
        )
        # check the bounds
        if (
            not np.all(self.l <= self.u)
            or not np.all(self.l <= self.n)
            or not np.all(self.n <= self.u)
        ):
            err = max(
                np.max(self.l - self.u),
                np.max(self.l - self.n),
                np.max(self.n - self.u),
            )
            LOGGER.warning("Invalid bounds for MVar, violated by " + str(err))
            # raise ValueError("Invalid bounds for MVar.")
        if not self.l.shape == self.n.shape == self.u.shape == self.v.shape:
            raise ValueError("Shape mismatch for MVar bounds.")
        # add the MIPStart
        if isinstance(self.v, gp.MVar):
            self.v.Start = self.n  # type: ignore

    @property
    def X(self) -> np.ndarray | float:  # noqa
        """Return the current value of the variable."""
        if isinstance(self.v, gp.MVar):
            return self.v.X
        if isinstance(self.v, gp.MLinExpr | gp.MQuadExpr):
            return self.v.getValue()
        if isinstance(self.v, np.ndarray):
            return self.v
        raise ValueError("Invalid variable type.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the variable."""
        return self.n.shape

    def __getitem__(self, key: int | slice | tuple[int, ...]) -> BoundedVar:
        """Return a subvariable."""
        return BoundedVar(
            n=self.n[key], l=self.l[key], u=self.u[key], v=self.v[key]
        )

    def reshape(self, shape: tuple[int, ...]) -> BoundedVar:
        """Return a reshaped variable."""
        return BoundedVar(
            n=self.n.reshape(shape),
            l=self.l.reshape(shape),
            u=self.u.reshape(shape),
            v=self.v.reshape(shape),  # type: ignore
        )

    @property
    def T(self) -> BoundedVar:  # noqa
        """Return the transpose of the variable."""
        return BoundedVar(n=self.n.T, l=self.l.T, u=self.u.T, v=self.v.T)  # type: ignore
