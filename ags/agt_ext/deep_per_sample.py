"""Per-sample parameter gradients for stacks with more than one trainable layer.

WRITTEN AS AN ADDITION, NOT AN EDIT. Nothing in abstract_gradient_training_opt is
modified; this file composes its public primitives and rebinds two methods inside a
scoped context manager, in the same style as agt_ext/clipped_disjoint.py.

--------------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------------
The clipping-site refactor (commit bb538c79, "certify the algorithm AGS actually
runs") moved per-sample clipping from dL/dlogit to dL/dtheta, which is where
propagate_clipping -- and therefore AGS, IBP and DP-SGD -- applies it. To do that it
materialises per-sample parameter gradients before the batch contraction.

That materialisation is only implemented for a layer whose INPUT IS CONSTANT
(functional.py:619-622). For the first Linear layer the input is the observed data,
constant when input_domain is None, so dl_dw[i,o,j] = dl_dy[i,o] * x[i,j] is linear.
For any deeper Linear layer the input is relu(W_1 x + b_1); the data is still
constant but W_1, b_1 are decision variables, so the outer product is BILINEAR and
functional.add_per_sample_param_grads raises rather than return a weaker bound.
MIPSequential.get_parameter_grads_per_sample therefore guards with
"support exactly one trainable layer".

The guard's stated reason -- that the module chain does not carry per-sample dl_dx --
is not the operative one: MIPSequential.add_backward already backpropagates dl_dy
through the reversed chain and every MIPLinear tries to build its own per-sample
grads from it. The operative gate is the constant-input test alone.

The missing piece already exists in this package: clipped_disjoint.per_sample_param_grads
handles a variable input with the four McCormick envelopes, which are the tightest
linear relaxation of a bilinear term over a box. This file routes deeper layers to it.

Soundness: McCormick is an OUTER relaxation, so the per-sample interval contains the
true product; clamping and the batch sum are monotone, so the resulting parameter
bounds stay valid and are at worst looser than an exact bilinear encoding. NonConvex=2
is already set by the callers, so an exact encoding is available as a later tightening.

Note also that at the FIRST step the parameters are still the fixed initialisation, so
a deeper layer's input is constant there too and takes the linear branch; the McCormick
branch only engages from the second step onward.
"""

from __future__ import annotations

import contextlib
import itertools

import gurobipy as gp

from abstract_gradient_training_opt.formulations import functional
from abstract_gradient_training_opt.formulations.modules import MIPLinear, MIPSequential
from abstract_gradient_training_opt.formulations.variables import BoundedVar

from ags.agt_ext.clipped_disjoint import per_sample_param_grads as _mccormick

_VAR_TYPES = (gp.MVar, gp.MLinExpr, gp.MQuadExpr)


def _per_sample_grads_any(formulation, x, dl_dy, suffix=""):
    """Per-sample (dl_dw, dl_db), for a constant or a variable layer input.

    Constant x  -> functional.add_per_sample_param_grads  (exact equalities, upstream)
    Variable x  -> clipped_disjoint.per_sample_param_grads (McCormick envelopes)

    Both return shapes (B, out, in) and (B, out), so the caller cannot tell them apart.
    """
    if isinstance(x.v, _VAR_TYPES):
        return _mccormick(formulation, x, dl_dy, suffix=suffix)
    return functional.add_per_sample_param_grads(formulation, x, dl_dy, suffix=suffix)


def _linear_add_backward(self, dl_dy: BoundedVar, suffix: str = "") -> BoundedVar:
    """MIPLinear.add_backward with the constant-input gate removed.

    Identical to the original except that the per-sample parameter gradients are
    always built, via _per_sample_grads_any, instead of being left as None when the
    layer input is a gurobi expression.
    """
    if self.x is None:
        raise ValueError("MIPModule.add_forward must be called before add_backward.")
    dl_dx, dl_dw, dl_db = functional.add_linear_gradient_constr(
        self.formulation,
        self.x,
        self.w,
        dl_dy,
        linear_grad_encoding=self.backward_encoding,
        suffix=suffix,
    )
    self.dl_dw, self.dl_db = dl_dw, dl_db
    self.dl_dw_per, self.dl_db_per = _per_sample_grads_any(self.formulation, self.x, dl_dy, suffix=suffix)
    return dl_dx


def _seq_get_parameter_grads_per_sample(self) -> list[BoundedVar]:
    """Per-sample parameter grads concatenated over every trainable module.

    Forward module order, matching get_parameters() and get_parameter_grads(), so that
    step(gradient_override=...) receives them in the order its length check expects.
    """
    return list(
        itertools.chain.from_iterable(
            m.get_parameter_grads_per_sample() for m in self.modules if m.get_parameters()
        )
    )


@contextlib.contextmanager
def use_deep_per_sample():
    """Bind the two replacements for the duration; restore on exit."""
    orig_backward = MIPLinear.add_backward
    orig_getter = MIPSequential.get_parameter_grads_per_sample
    MIPLinear.add_backward = _linear_add_backward
    MIPSequential.get_parameter_grads_per_sample = _seq_get_parameter_grads_per_sample
    try:
        yield
    finally:
        MIPLinear.add_backward = orig_backward
        MIPSequential.get_parameter_grads_per_sample = orig_getter
