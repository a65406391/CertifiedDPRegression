"""Per-sample parameter gradients for the three 10-class arms under comparison.

Each function returns dL/dtheta per sample, shape (n, P) with P = C*d + C, flattened
in torch's Linear(d, C) order so the numpy trajectory and AGT's nominal trajectory
agree exactly from a zero init.

  cross_entropy   dL/dz = softmax(z) - onehot(y)
                  Matches BoundedCrossEntropyLoss.backward exactly.
                  CERTIFIABLE: AGT implements this loss.

  hinge_ova       one-vs-all. C independent binary hinges on t_j = +1 if j == y else
                  -1:  L = sum_j max(0, margin - t_j z_j),  dL/dz_j = -t_j 1[active].
                  Matches BoundedHingeLoss.backward applied elementwise to a
                  (n, C) target matrix of +-1.
                  CERTIFIABLE: AGT's hinge accepts a (n, C) +-1 target matrix -- it
                  checks only that every entry is +-1 and reshapes inputs to match.

  hinge_ww        Weston-Watkins multiclass hinge:
                  L = sum_{j != y} max(0, margin + z_j - z_y)
                  dL/dz_j = 1[active_j] for j != y ;  dL/dz_y = -sum_{j != y} 1[..]
                  NOT CERTIFIABLE TODAY: AGT has no multiclass hinge. BoundedHingeLoss
                  raises ValueError("Labels must be binary for hinge loss.") for any
                  target outside {0,1}/{-1,+1}, and there is no coupled-logit hinge
                  encoding in abstract_gradient_training_opt either. Included because
                  it is the textbook multiclass hinge and the nominal question asked
                  is which loss trains best -- but if it wins, certifying it is a
                  library extension, not a configuration change.
"""

import numpy as np

N_CLASSES = 10


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _to_param_grads(dz, X):
    """(n, C) logit grads + (n, d) inputs -> (n, C*d + C) parameter grads."""
    dW = dz[:, :, None] * X[:, None, :]  # (n, C, d), torch weight order
    return np.concatenate([dW.reshape(len(X), -1), dz], axis=1)


def cross_entropy(z, y, X, n_classes=N_CLASSES, margin=1.0):
    dz = _softmax(z)
    dz[np.arange(len(y)), y] -= 1.0
    return _to_param_grads(dz, X)


def hinge_ova(z, y, X, n_classes=N_CLASSES, margin=1.0):
    t = -np.ones((len(y), n_classes))
    t[np.arange(len(y)), y] = 1.0
    dz = -t * ((margin - t * z) > 0)
    return _to_param_grads(dz, X)


def hinge_ww(z, y, X, n_classes=N_CLASSES, margin=1.0):
    zy = z[np.arange(len(y)), y][:, None]
    active = (margin + z - zy) > 0
    active[np.arange(len(y)), y] = False
    dz = active.astype(float)
    dz[np.arange(len(y)), y] = -active.sum(1)
    return _to_param_grads(dz, X)


GRADS = {"cross_entropy": cross_entropy, "hinge_ova": hinge_ova, "hinge_ww": hinge_ww}

# Which arms abstract_gradient_training can bound. hinge_ww is measurable but not
# certifiable; anything selecting it for the certified path must fail loudly.
CERTIFIABLE = {"cross_entropy": "cross_entropy", "hinge_ova": "hinge"}
