"""The cross-radius certificate for the blobs release, beside the pinned reference.

`ss_reference.local_sens` takes the per-coordinate WIDTH of the k-envelope as A^k. The paper's
A^(k) is the largest one-substitution jump from any dataset within k of D, and from nested
envelopes T_k within T_(k+1) the tight bound is the cross-radius

    Abar_j(k) = max(U_j^(k+1) - L_j^k,  U_j^k - L_j^(k+1)),

with Abar_j(0) the radius of the 1-box about the nominal. Since W(k) <= Abar(k) <= W(k+1), the
reference is optimistic at every computed radius and conservative at k = 0, where it uses W(1)
in place of the radius. On the stored blobs envelopes the two disagree by at most a factor
e^(beta/P) at any radius; the accuracy shift is tabulated in Notes/REFACTOR_LOG.md.

Selected by `--certificate cross_radius` on plot_headline, plot_approx and plot_combined. The
default stays the reference, so every stored figure and JSON is untouched; this path writes its
outputs with a `_cross_radius` suffix. The mathematics is `common.certificate`, the same code
AGT-R releases with; this module only adapts it to the blobs conventions (the beta/P split, the
GS cap, the stored box layout).
"""

import numpy as np

from common import certificate


def _as_boxes(boxes):
    return {
        int(k): (np.asarray(lo, float).ravel(), np.asarray(hi, float).ravel())
        for k, (lo, hi) in boxes.items()
    }


def certificate_grid(boxes, theta_n, gs_vec):
    """Abar_j(k) for k = 0 .. k_max, shape (k_max + 1, P), capped at GS.

    `boxes` maps each computed radius k >= 1 to (param_l, param_u). Radii between computed ones
    are back-filled with the next computed envelope and the sequence is nested, both of which are
    sound containments. T_(k_max+1) is not stored and is taken at the GS cap, which is sound because
    a one-substitution jump never exceeds GS.
    """
    theta_n = np.asarray(theta_n, float).ravel()
    gs_vec = np.asarray(gs_vec, float).ravel()
    return certificate.cross_radius(
        theta_n,
        _as_boxes(boxes),
        tail=(theta_n - gs_vec, theta_n + gs_vec),
        cap=gs_vec,
        warn_on_gaps=False,  # reported once by `report`, not on every epsilon
    )


def smooth_sensitivity(boxes, theta_n, gs_vec, beta, n_params):
    """Per-coordinate SS_j under the cross-radius certificate.

    Drop-in for `ss_reference.smooth_sensitivity`: the same beta / n_params split (reference L437)
    and the same discounted GS tail, with Abar(k) in place of min(W(k), GS * k).
    """
    A = certificate_grid(boxes, theta_n, gs_vec)
    return certificate.smooth_sensitivity(
        A, beta / float(n_params), tail_width=np.asarray(gs_vec, float).ravel()
    )


def report(boxes, theta_n, gs_vec):
    """W(k) against Abar(k), mean over coordinates, and which radii were back-filled."""
    b = _as_boxes(boxes)
    A = certificate_grid(b, theta_n, gs_vec)
    ks = sorted(b)
    gaps = sorted(set(range(1, ks[-1] + 1)) - set(ks))
    lines = [
        f"  certificate: cross-radius; computed radii {ks}; back-filled {gaps or 'none'}; "
        f"GS = {float(np.mean(gs_vec)):.4f}; Abar(0) = {A[0].mean():.4f} (radius of the 1-box)",
        f"  {'k':>4} {'W(k)':>9} {'Abar(k)':>9} {'Abar/W':>7}",
    ]
    for k in ks:
        w = float(np.abs(b[k][1] - b[k][0]).mean())
        lines.append(f"  {k:>4} {w:9.4f} {A[k].mean():9.4f} {A[k].mean() / max(w, 1e-12):7.3f}")
    return "\n".join(lines)
