"""The single grid both arms are tuned over. Symmetry is enforced by sharing this file.

lambda = 0 throughout: regularisation is deliberately absent here, a load-bearing feature
of the setting rather than a knob. gamma is fixed at the headline value because the clip
is not independently free -- it sets how many updates a run can afford -- so it is pinned
rather than searched. The free axes are the training-run parameters (lr, batch, epochs);
the noise, and hence what each eps buys, follows from them.
"""

EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100]
BS = [200, 100, 50, 25, 20, 10]
EPOCHS = [1, 2, 5, 10, 20]
LRS = [0.1, 0.25, 0.5]
GAMMA = 0.05
LAM = 0.0


def grid():
    """(b, epochs, lr) triples, cheap-first: AGS bound cost scales with n_s."""
    g = [(b, ep, lr) for b in BS for ep in EPOCHS for lr in LRS]
    g.sort(key=lambda t: t[1] * max(1, 200 // t[0]))
    return g
