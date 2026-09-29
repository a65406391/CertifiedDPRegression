"""Validation for AGS_Ablation/agt_ext/deep_per_sample.py.

(i)  single trainable layer: use_deep_per_sample must reproduce the default path
     bit-identically -- it touches nothing there.
(ii) 8->16->1: the certified envelope must contain the nominal trajectory, and must
     not be tighter than the IBP bound the same run tracks.
"""

import warnings

warnings.simplefilter("ignore")
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from common.data import load_california_data
from common.models import create_mlp
from abstract_gradient_training_opt.certified_training import certified_training
from abstract_gradient_training_opt.configuration import OPTConfig
from ags.agt_ext.deep_per_sample import use_deep_per_sample

B, NDATA, LR, CLIP, K, F, SEED = 32, 64, 0.1, 0.1, 2, 2, 42


def data():
    tr, *_ = load_california_data()
    X = tr.tensors[0][:NDATA].double()
    Y = tr.tensors[1][:NDATA].double()
    return TensorDataset(X, Y), X, Y


def cfg():
    return OPTConfig(
        log_level="ERROR",
        n_epochs=1,
        batchsize=B,
        learning_rate=LR,
        loss="mse",
        concretization_freq=F,
        k_substitution=K,
        input_domain=None,
        label_domain=None,
        gradient_clipping=CLIP,
        relax_binaries=False,
        relax_bilinear=False,
        gurobi_kwargs={"NonConvex": 2, "OutputFlag": 0, "Threads": 8, "TimeLimit": 120},
        metadata={"seed": SEED, "strict_mode": True},
    )


def run(hidden, deep):
    ds, X, Y = data()
    torch.manual_seed(SEED)
    m = create_mlp(8, [] if hidden == 0 else [hidden]).double()
    if deep:
        with use_deep_per_sample():
            out, _ = certified_training(m, cfg(), ds, solve=True)
    else:
        out, _ = certified_training(m, cfg(), ds, solve=True)
    f = lambda ps: np.concatenate([p.detach().cpu().numpy().ravel() for p in ps])
    return f(out.param_l), f(out.param_n), f(out.param_u)


def nominal_trajectory(hidden):
    """The algorithm the MIP encodes, run on the unmodified data.

    dl_dlogit_i = 2 (yhat_i - y_i)                   functional.py:2210
    per-sample parameter grad, clipped ELEMENTWISE at gamma, THEN summed,
    then divided by the batch size inside step().    modules.py:392-397

    Clipping the summed gradient instead of each sample's is a different algorithm
    and does not have to lie inside the envelope.
    """
    from torch.func import functional_call, grad, vmap

    ds, X, Y = data()
    torch.manual_seed(SEED)
    m = create_mlp(8, [] if hidden == 0 else [hidden]).double()
    params = {k: v.detach().clone() for k, v in m.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in m.named_buffers()}

    def per_sample_loss(pr, bu, x, y):
        out = functional_call(m, (pr, bu), (x.unsqueeze(0),))
        return ((out.squeeze() - y.squeeze()) ** 2).sum()

    gfn = vmap(grad(per_sample_loss), in_dims=(None, None, 0, 0))
    for xb, yb in DataLoader(ds, batch_size=B, shuffle=False):
        g = gfn(params, buffers, xb, yb)
        with torch.no_grad():
            for k in params:
                params[k] = params[k] - LR * g[k].clamp(-CLIP, CLIP).sum(0) / B
    return np.concatenate([params[k].detach().numpy().ravel() for k, _ in m.named_parameters()])


if __name__ == "__main__":
    print("(i) single trainable layer, Linear(8,1): default vs use_deep_per_sample")
    l0, n0, u0 = run(0, deep=False)
    l1, n1, u1 = run(0, deep=True)
    same = np.array_equal(l0, l1) and np.array_equal(n0, n1) and np.array_equal(u0, u1)
    print(
        "    max |dl|=%.3e  max |du|=%.3e  max |dn|=%.3e"
        % (np.abs(l0 - l1).max(), np.abs(u0 - u1).max(), np.abs(n0 - n1).max())
    )
    print("    bit-identical: %s" % same)
    assert same, "(i) FAILED: the extension changed a single-layer result"

    print("\n(ii) 8->16->1 ReLU")
    l, n, u = run(16, deep=True)
    ok_order = bool(np.all(l <= n + 1e-12) and np.all(n <= u + 1e-12))
    print("    P=%d  envelope l<=n<=u: %s" % (l.size, ok_order))
    print("    width  median=%.4e  max=%.4e  min=%.4e" % (np.median(u - l), (u - l).max(), (u - l).min()))
    assert ok_order, "(ii) FAILED: envelope does not contain its own nominal"

    theta = nominal_trajectory(16)
    inside = bool(np.all(theta >= l - 1e-9) and np.all(theta <= u + 1e-9))
    viol = np.maximum(l - theta, theta - u).max()
    print("    nominal trajectory inside envelope: %s (max violation %.3e)" % (inside, viol))
    assert inside, "(ii) FAILED: unmodified-data trajectory falls outside the envelope"
    print("\nALL CHECKS PASSED")
