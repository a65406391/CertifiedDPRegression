"""
Abstract Gradient Sampling (AGS) — private learning via windowed AGT.

AGS partitions n_total training steps into windows of size s_p.  Within each
window the standard AGT privacy-certified loop runs (with k_private=1),
producing parameter envelopes [θ_L, θ_U].  At the window boundary we:

        1. Compute per-coordinate smooth sensitivity from the envelopes.
        2. Add independent Cauchy noise calibrated to the smooth sensitivity.
        3. Reset the parameter bounds to the privatised point.

Each window release is ε-DP, giving (ε · n_windows)-DP overall by sequential
composition of pure DP.

This module also contains a Laplace DP-SGD baseline that adds Laplace noise
to each gradient step (with per-element clipping), for direct comparison.

Oracle noise mode (optional)
----------------------------
When a NoiseDecisionTable is provided, the privatisation step selects
Cauchy or Laplace noise *per parameter tensor* based on the one-step
dominance condition.  This is a non-private oracle analysis — consulting
data-dependent SS values breaks DP.  See common.noise_decision.
"""

import copy
import gc
import logging
import math
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.paths import get_repo_root

logger = logging.getLogger(__name__)


def setup_file_logger(
    log_file: Optional[Union[str, Path]] = None,
    dataset: Optional[str] = None,
    privacy_mode: Optional[str] = None,
    exp_type: Optional[str] = None,
    log_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Configure a FileHandler for the root and module loggers so outputs are saved to a file.

    File naming format by default:
        <repo_root>/logs/{dataset}_{pure/approx_dp}_{sampling/performance/hybrid}.log
    """
    if log_file is not None:
        log_path = Path(log_file)
    else:
        ds = dataset or "mnist"
        pm = "approx_dp" if (privacy_mode == "approx" or (privacy_mode and "approx" in str(privacy_mode))) else "pure_dp"
        exp = exp_type or "performance"
        base_dir = Path(log_dir) if log_dir else get_repo_root() / "logs"
        log_path = base_dir / f"{ds}_{pm}_{exp}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path.resolve():
            return log_path

    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(levelname)-8s] %(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logger.info("Logging to file: %s", log_path)
    return log_path


from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_models import (
    CROWNBoundedModel,
    IntervalBoundedModel,
)
from abstract_gradient_training.bounded_models.mip_bounded_model import MIPBoundedModel
from abstract_gradient_training.certified_training import (
    PrivacyGradientAccumulator,
    training_utils,
)
from abstract_gradient_training.configuration import AGTConfig

if TYPE_CHECKING:
    from common.noise_decision import NoiseDecisionTable

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Smooth sensitivity helpers (parameter space)
# ---------------------------------------------------------------------------


def _param_envelope_width(bounded_model: IntervalBoundedModel):
    """Return per-coordinate width |θ_U - θ_L| as a flat list of tensors."""
    widths = []
    for pl, pu in zip(bounded_model.param_l, bounded_model.param_u):
        widths.append((pu - pl).abs())
    return widths


def _log_local_sens_diagnostics(local_sens, true_gs_window, k_subset, win_label=""):
    """Log per-k local sensitivity statistics: min, max, mean, and tightening vs GS."""
    for k in k_subset:
        ls_flat = torch.cat([t.flatten() for t in local_sens[k]])
        gs_flat = torch.cat([g.flatten() for g in true_gs_window]) * k
        n_total = ls_flat.numel()
        # Coordinates where local_sens == GS*k (bounds didn't tighten)
        at_gs = (ls_flat >= gs_flat - 1e-12).sum().item()
        under_gs = n_total - at_gs
        ratio = ls_flat / gs_flat.clamp(min=1e-30)
        logger.info(
            f"  {win_label}k={k}: local_sens  "
            f"mean={ls_flat.mean().item():.4e}  std={ls_flat.std().item():.4e}  "
            f"min={ls_flat.min().item():.4e}  max={ls_flat.max().item():.4e}  "
            f"at_GS={at_gs}/{n_total} ({100*at_gs/n_total:.1f}%)  "
            f"tightened={under_gs}/{n_total} ({100*under_gs/n_total:.1f}%)  "
            f"avg_ratio_to_GS={ratio.mean().item():.4f}"
        )


def _log_multiplier_diagnostics(beta_js, ss, a_js, sigma_js, label=""):
    """Log concise stats for the privacy multipliers."""
    def _stats(tensors):
        flat = torch.cat([t.flatten() for t in tensors])
        return flat.mean().item(), flat.std().item(), flat.min().item(), flat.max().item()

    bm, bs, bmin, bmax = _stats(beta_js)
    logger.info(
        f"  {label}beta_j (SS decay rate per param, sum=beta):  "
        f"mean={bm:.4e}  std={bs:.4e}  min={bmin:.4e}  max={bmax:.4e}"
    )
    sm, ss_s, smin, smax = _stats(ss)
    logger.info(
        f"  {label}SS (smooth sensitivity, max over staircase):  "
        f"mean={sm:.4e}  std={ss_s:.4e}  min={smin:.4e}  max={smax:.4e}"
    )
    am, as_, amin, amax = _stats(a_js)
    logger.info(
        f"  {label}a_j (budget allocation weight, sum(1/a_j)=1):  "
        f"mean={am:.4e}  std={as_:.4e}  min={amin:.4e}  max={amax:.4e}"
    )
    sgm, sgs, sgmin, sgmax = _stats(sigma_js)
    logger.info(
        f"  {label}sigma_j (noise scale = C/eps * SS * a_j):  "
        f"mean={sgm:.4e}  std={sgs:.4e}  min={sgmin:.4e}  max={sgmax:.4e}"
    )


# ---------------------------------------------------------------------------
# Infinite dataloader iterator
# ---------------------------------------------------------------------------


def _infinite_loader(loader: DataLoader):
    """Yield batches from *loader* in an infinite cycle."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# AGS Training
# ---------------------------------------------------------------------------


_DEFAULT_K_SUBSET = [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]


@torch.no_grad()
def train_ags(
    model: nn.Sequential,
    train_loader: DataLoader,
    *,
    n_total_steps: int,
    sampling_freq: int,
    concretization_freq: int,
    epsilon_per_step: float,
    lr: float,
    clip_gamma: float,
    loss: str = "cross_entropy",
    bound_mode: str = "ibp",
    device: str = "cpu",
    seed: int = 42,
    eta: float = 1.0,
    fragsize: int = 1,
    oracle_noise_table: Optional["NoiseDecisionTable"] = None,
    eval_callback: Optional[callable] = None,
    k_subset: Optional[List[int]] = None,
    privacy_mode: str = "pure",
    delta: Optional[float] = None,
    save_bounds_path: Optional[str] = None,
    gamma_cauchy: int = 2,
    noise_mode: str = "anisotropic", # in {anisotropic, isotropic}
    log_file: Optional[Union[str, Path]] = None,
    dataset: Optional[str] = None,
    exp_type: Optional[str] = None,
):
    """Train a model with Abstract Gradient Sampling.

    Args:
            model: A fresh nn.Sequential model (will be copied internally).
            train_loader: DataLoader for the training set.
            n_total_steps: Total number of gradient update steps.
            sampling_freq: Window size s_p (steps between privatisations).
            concretization_freq: Number of steps the MILP computes the bound (if MILP is chosen).
            epsilon_per_step: Privacy budget ε per training step (total = eps * n_steps).
            lr: Learning rate.
            clip_gamma: Per-element gradient clipping parameter.
            loss: Loss function name (must match AGTConfig choices).
            bound_mode: How to compute bounds ("ibp", "miqp", "milp", etc.).
            device: Device string.
            seed: Random seed.
            eta: Learning rate decay parameter for AGS (lr_n = lr / (1 + eta * n)).
            oracle_noise_table: Optional NoiseDecisionTable for per-parameter
                Cauchy vs Laplace selection.  When provided, each parameter
                tensor independently chooses the noise mechanism that gives
                the tighter high-confidence bound.  **Breaks DP** — oracle only.
            k_subset: Finite subset of k values for the staircase smooth
                sensitivity computation.  If None, defaults to
                ``_DEFAULT_K_SUBSET``.
            privacy_mode: ``"pure"`` for (ε,0)-DP with Cauchy noise, or
                ``"approx"`` for (ε,δ)-DP with Laplace noise on smooth
                sensitivity.
            delta: Privacy δ for approximate DP.  Required when
                ``privacy_mode == "approx"``.
            save_bounds_path: If provided, save the per-k upper/lower parameter
                bounds, nominal parameters, and global sensitivity to this
                ``.pt`` file.  Enables fast re-evaluation at different ε
                via :func:`ags_inference_from_bounds`.
            log_file: Optional path to log file.
            dataset: Dataset name for automatic log file naming.
            exp_type: Experiment type ('performance', 'sampling', 'hybrid').

    Returns:
            model_out: nn.Sequential with privatised parameters.
            info: Dict with per-window diagnostics (smooth sensitivity, noise, …).
    """
    if log_file is not None or dataset is not None or exp_type is not None:
        setup_file_logger(
            log_file=log_file,
            dataset=dataset,
            privacy_mode=privacy_mode,
            exp_type=exp_type,
        )

    if privacy_mode not in ("pure", "approx"):
        raise ValueError(f"privacy_mode must be 'pure' or 'approx', got {privacy_mode!r}")
    if privacy_mode == "approx" and delta is None:
        raise ValueError("delta must be provided when privacy_mode='approx'")

    if gamma_cauchy != 2:
        raise ValueError("Generalized Cauchy for gamma != 2 has no closed from and is thus not implemented yet.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if k_subset is None:
        k_subset = list(_DEFAULT_K_SUBSET)
    k_subset = sorted(set(k_subset))  # ensure sorted and unique

    # Deep-copy so the caller's model is untouched
    model = copy.deepcopy(model).to(device)

    if bound_mode == "ibp":
        bounded_model = IntervalBoundedModel(model)
    elif bound_mode == "crown":
        bounded_model = CROWNBoundedModel(
            model,
            gradient_bound_mode="linear",
        )
    else:
        bounded_model = MIPBoundedModel(
            model,
            forward_bound_mode=bound_mode,
            optimize_inter=True,
            concretization_frequency=concretization_freq,
            gurobi_kwargs={"OutputFlag": 0, "Threads": 16, "TimeLimit": 60},
        )
    bounded_model.to(device)

    # Precompute part of GS bound (will be multiplied by sum of LRs)
    first_batch = next(iter(train_loader))[0]
    batchsize = first_batch.size(0)

    data_iter = _infinite_loader(train_loader)

    n_windows = math.ceil(n_total_steps / sampling_freq)
    window_info = []

    n_params = sum(p.numel() for p in bounded_model.param_n)
    step = 0
    for win_idx in range(n_windows):
        window_steps = min(sampling_freq, n_total_steps - step)

        # Total privacy budget for this specific window
        epsilon_per_window = epsilon_per_step * window_steps

        # Set β depending on privacy mode - the same across windows
        if privacy_mode == "pure":
            beta = epsilon_per_window / (2.0 * (gamma_cauchy + 1.0)) -1e-7 # epsilon / 6 when anisotropic and gamma_cauchy = 2
        else:  # approx
            # beta = epsilon_per_window / (2.0 * math.log(2.0 / delta)) - 1e-7
            beta = min(1.0, epsilon_per_window / (1.0 + 2 * np.e * float(np.log(n_params / delta))))

        # Save model state at the start of the window
        start_n = [p.detach().clone() for p in bounded_model.param_n]
        start_l = [p.detach().clone() for p in bounded_model.param_l]
        start_u = [p.detach().clone() for p in bounded_model.param_u]

        # Cache the batches for this window
        window_batches = []
        for _s in range(window_steps):
            window_batches.append(next(data_iter))

        # Compute total sum of learning rates over this window
        sum_lr_window = sum(lr / (1 + eta * (step + s_idx)) for s_idx in range(window_steps))
        # The true Global Sensitivity over this window (for neighboring datasets, k=1)
        true_gs_window = [torch.full_like(p, sum_lr_window * 2 * clip_gamma / batchsize) for p in bounded_model.param_n]

        # ---- Staircase smooth sensitivity computation ----
        # 1. Initialize SS to global sensitivity for all coordinates
        ss = [gs.clone() for gs in true_gs_window]

        # 2. Compute local sensitivity A^k for each k in k_subset
        local_sens = {}  # k -> list of tensors (one per parameter)
        _per_k_bounds = {}  # k -> {'param_l': [...], 'param_u': [...]} for saving

        # Estimate: AGS runs the full window training for each k in k_subset using bounded arithmetic.
        # Bounded arithmetic (e.g. CROWN) is typically ~2-5x slower than standard forward/backward.
        # So AGS is roughly `len(k_subset) * bounded_overhead` times slower than DP-SGD.
        for k in tqdm(k_subset, desc=f"AGS Window {win_idx+1}/{n_windows} (k-search)", leave=False):
            gs_window_k = [gs * k for gs in true_gs_window]

            # Reset model to start of window before re-running
            for p, start_p in zip(bounded_model.param_n, start_n):
                p.copy_(start_p)
            for p, start_p in zip(bounded_model.param_l, start_l):
                p.copy_(start_p)
            for p, start_p in zip(bounded_model.param_u, start_u):
                p.copy_(start_p)

            # Run window_steps AGT gradient updates for this k
            for s_idx, (batch, labels) in tqdm(enumerate(window_batches), total=window_steps, desc=f"k={k} Steps", leave=False):
                lr_n = lr / (1 + eta * (step + s_idx))

                config = AGTConfig(
                    n_epochs=1,
                    learning_rate=lr_n,
                    loss=loss,
                    k_private=k,
                    clip_gamma=clip_gamma,
                    clip_method="clamp",
                    device=device,
                    log_level="WARNING",
                    fragsize=batchsize,
                )

                optimizer = config.get_bounded_optimizer(bounded_model)
                gradient_accumulator = PrivacyGradientAccumulator(
                    config.k_private,
                    config.clip_gamma,
                    bounded_model.param_n,
                )

                batch, labels = batch.to(device), labels.to(device)

                # Loop over fragments within the batch for correct per-sample clipping
                frag_size = config.fragsize
                for i in range(0, batch.size(0), frag_size):
                    f_batch = batch[i : i + frag_size]
                    f_labels = labels[i : i + frag_size]

                    f_grads_n = training_utils.compute_batch_gradients(bounded_model, f_batch, f_labels, config, nominal=True)
                    f_grads_l, f_grads_u = training_utils.compute_batch_gradients(bounded_model, f_batch, f_labels, config, nominal=False)
                    f_grads_l, f_grads_n, f_grads_u = training_utils.propagate_clipping(
                        f_grads_l, f_grads_n, f_grads_u, config.clip_gamma, config.clip_method
                    )
                    gradient_accumulator.add_private_fragment_gradients(f_grads_n, f_grads_l, f_grads_u)

                update_l, update_n, update_u = gradient_accumulator.concretize_gradient_update(batch.size(0))
                optimizer.step(update_l, update_n, update_u)

            # Local sensitivity A^k = min(envelope_width, GS * k)
            widths = _param_envelope_width(bounded_model)
            local_sens[k] = [torch.minimum(w, g) for w, g in zip(widths, gs_window_k)]

            # Save k=1 nominal params (pre-noise) for bounds file
            if k == 1:
                _k1_param_n = [p.detach().cpu().clone() for p in bounded_model.param_n]

            # Capture per-k bounds for offline re-evaluation
            if save_bounds_path is not None:
                _per_k_bounds[k] = {
                    'param_l': [p.detach().cpu().clone() for p in bounded_model.param_l],
                    'param_u': [p.detach().cpu().clone() for p in bounded_model.param_u],
                }

        _log_local_sens_diagnostics(local_sens, true_gs_window, k_subset, f"[Window {win_idx+1}/{n_windows}] ")

        # 3. Build staircase: for integer k in [0, max_k_subset], assign
        #    local sensitivity from the *next* computed k in subset.
        #    Between k_{i} and k_{i+1}, use A^{k_{i+1}}.
        #    Before k_subset[0], use A^{k_subset[0]}.
        max_k_val = k_subset[-1]

        # We only need SS at each k_val in {0, 1, ..., max_k_val},
        # but we build the staircase lazily: for each integer k,
        # staircase_A(k) = A^{k_j} where k_j is the smallest element
        # of k_subset that is >= k.
        # Then SS = max_k { e^{-beta*k} * min(staircase_A(k), GS) }
        #
        # Since e^{-beta*k} decays, we only need to check at the
        # transition points of the staircase (i.e. at k_subset values
        # and at k=0).

        # Reset SS to initialization (GS) — we will take min with staircase
        ss = [gs.clone() for gs in true_gs_window]
        # set beta_js as per-layer list matching the structure of ss
        # Uniform initialisation: beta_j = beta / d for every coordinate
        beta_js = [torch.full_like(p, beta / n_params) for p in bounded_model.param_n]
        if noise_mode == "anisotropic":
            # Compute growth rates from flattened local_sens, then split back to per-layer
            local_sens_flatten_per_k = {k: torch.cat([ls.flatten() for ls in local_sens[k]]) for k in k_subset}
            growth_rates_flat = torch.zeros(n_params, device=device)
            for k, km1 in zip(k_subset[1:], k_subset[:-1]):
                ls_k = local_sens_flatten_per_k[k].clamp(min=1e-30)
                ls_km1 = local_sens_flatten_per_k[km1].clamp(min=1e-30)
                growth_rates_flat += torch.log(ls_k / ls_km1)
            growth_rates_flat *= (1.0 / len(k_subset))
            # Split growth_rates back to per-layer and scale beta_js
            layer_numels = [p.numel() for p in bounded_model.param_n]
            gr_layers = list(growth_rates_flat.split(layer_numels))
            beta_js = [bj * gr.view(bj.shape) for bj, gr in zip(beta_js, gr_layers)]
            # Renormalise so that sum of all beta_j = beta
            beta_sum = sum(bj.sum().item() for bj in beta_js)
            beta_js = [beta * bj / beta_sum for bj in beta_js]

            logger.info(
                f"  [Window {win_idx+1}/{n_windows}] Anisotropic Growth Rates: "
                f"mean={growth_rates_flat.mean().item():.4e}, std={growth_rates_flat.std().item():.4e}, "
                f"min={growth_rates_flat.min().item():.4e}, max={growth_rates_flat.max().item():.4e}"
            )


        # For each k in k_subset and k=0 (the staircase transition points),
        # compute the smooth sensitivity candidate and take the max.
        # At k=0: staircase value = A^{k_subset[0]}, decay = e^0 = 1
        # At k_subset[j]: staircase value = A^{k_subset[j]}, decay = e^{-beta*k_subset[j]}
        #
        # For k between k_subset[j-1]+1 and k_subset[j], the staircase
        # assigns A^{k_subset[j]}, and the decay is e^{-beta*k}. The
        # maximum over this range occurs at k = k_subset[j-1]+1 (smallest k,
        # largest decay). But since k_subset[j-1]+1 has the same A as
        # k_subset[j], we check at k_subset[j-1]+1 with A^{k_subset[j]}.

        # Collect all check-points: {0} ∪ {k+1 for k in k_subset[:-1]} ∪ k_subset
        check_points = set()
        check_points.add(0)
        for kv in k_subset:
            check_points.add(kv)
        for j in range(len(k_subset) - 1):
            check_points.add(k_subset[j] + 1)
        check_points = sorted(check_points)

        # Map each check-point to the staircase local sensitivity
        def _get_staircase_ls(k_val):
            """Return the staircase local sensitivity tensors for integer k_val."""
            # Find the smallest k in k_subset >= k_val
            for kv in k_subset:
                if kv >= k_val:
                    return local_sens[kv]
            # k_val > max of k_subset: use GS * k_val as fallback
            return [gs * k_val for gs in true_gs_window]

        # Now compute smooth sensitivity as max over check-points
        # decay is per-coordinate: decay_j(k) = exp(-beta_j * k)
        ss = [torch.zeros_like(p) for p in bounded_model.param_n]
        max_ss_sum = -1.0
        dominant_cp = -1

        for cp in check_points:
            staircase_ls = _get_staircase_ls(cp)
            cp_sum = 0.0
            for i in range(len(ss)):
                decay = torch.exp(-beta_js[i] * cp)  # per-coordinate decay
                capped = torch.minimum(staircase_ls[i], true_gs_window[i])
                candidate = decay * capped
                cp_sum += candidate.sum().item()
                ss[i] = torch.maximum(ss[i], candidate)

            if cp_sum > max_ss_sum:
                max_ss_sum = cp_sum
                dominant_cp = cp

        # Also check the tail: for k > max_k_val, GS * e^{-beta_j*k} is
        # the bound. The maximum of this tail is at k = max_k_val + 1.
        tail_sum = 0.0
        for i in range(len(ss)):
            tail_decay = torch.exp(-beta_js[i] * (max_k_val + 1))
            tail_candidate = tail_decay * true_gs_window[i]
            tail_sum += tail_candidate.sum().item()
            ss[i] = torch.maximum(ss[i], tail_candidate)

        if tail_sum > max_ss_sum:
            dominant_cp = max_k_val + 1

        step += window_steps
        ss_flat = torch.cat([s.flatten() for s in ss])
        ss_mean = ss_flat.mean().item()
        ss_std = ss_flat.std().item()

        # Compute allocation weights a_j and noise scales sigma_j (all per-layer lists)
        if noise_mode == "anisotropic":
            ss_sqrt_sum = sum(s.sqrt().sum().item() for s in ss)
            a_js = [ss_sqrt_sum / s.sqrt().clamp(min=1e-30) for s in ss]
        else:
            a_js = [torch.full_like(s, float(n_params)) for s in ss]
        s_js = [s * a for s, a in zip(ss, a_js)]
        if privacy_mode == "approx":
            sigma_js = [(2.0 / epsilon_per_window) * sj for sj in s_js]
        else:  # pure
            sigma_js = [((2.0 * gamma_cauchy) / epsilon_per_window) * sj for sj in s_js]

        _log_multiplier_diagnostics(beta_js, ss, a_js, sigma_js, f"[Window {win_idx+1}/{n_windows}] ")

        # Add independent noise per parameter tensor
        n_cauchy = 0
        n_laplace = 0
        for i, (pn, pl, pu) in enumerate(zip(bounded_model.param_n, bounded_model.param_l, bounded_model.param_u)):

            noise_scale = torch.clamp(sigma_js[i], min=1e-8)
            if privacy_mode == "approx":
                noise = torch.distributions.Laplace(
                    torch.zeros_like(pn), noise_scale
                ).sample()
                n_laplace += pn.numel()
            else:
                noise = torch.distributions.Cauchy(
                    torch.zeros_like(pn), noise_scale
                ).sample()
                n_cauchy += pn.numel()

            pn.add_(noise)
            pl.copy_(pn)
            pu.copy_(pn)

        avg_noise_scale = sum(sj.mean().item() for sj in sigma_js) / len(sigma_js)

        mode_str = f"  mode={privacy_mode}"
        if privacy_mode == "approx":
            mode_str += f"  delta={delta:.2e}"
        logger.info(
            f"  AGS window {win_idx + 1}/{n_windows}  SS_mean={ss_mean:.2e} (std={ss_std:.2e})  "
            f"dominant_k={dominant_cp}  "
            f"avg_noise={avg_noise_scale:.2e}  mode={privacy_mode}" + (f"  delta={delta:.2e}" if privacy_mode == "approx" else "")
        )

        window_info.append(
            {
                "window": win_idx,
                "steps": window_steps,
                "ss_mean": ss_mean,
                "ss_std": ss_std,
                "avg_noise_scale": avg_noise_scale,
                "n_cauchy": n_cauchy,
                "n_laplace": n_laplace,
                "privacy_mode": privacy_mode,
                "beta": beta,
            }
        )

        if eval_callback is not None:
            # Temporary model extraction for evaluation
            tmp_model = _extract_model(bounded_model, model)
            eval_callback(tmp_model, step)

    model_out = _extract_model(bounded_model, model)
    info = {
        "windows": window_info,
        "n_windows": n_windows,
        "k_subset": k_subset,
        "privacy_mode": privacy_mode,
    }
    if privacy_mode == "approx":
        info["delta"] = delta

    # Persist bounds for inference-only re-evaluation at different epsilon
    if save_bounds_path is not None:
        bounds_data = {
            'param_n': _k1_param_n,  # pre-noise, k=1 trained params
            'per_k_bounds': _per_k_bounds,
            'true_gs_window': [g.detach().cpu().clone() for g in true_gs_window],
            'k_subset': k_subset,
            'privacy_mode': privacy_mode,
            'n_windows': n_windows,
            'batchsize': batchsize,
            'clip_gamma': clip_gamma,
        }
        if privacy_mode == "approx":
            bounds_data['delta'] = delta
        torch.save(bounds_data, save_bounds_path)
        logger.info(f"Saved AGS bounds to {save_bounds_path}")

    return model_out, info





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_model(bounded_model: IntervalBoundedModel, template: nn.Sequential):
    """Copy privatised parameters from *bounded_model* back into a Sequential."""
    state = template.state_dict()
    flat_params = bounded_model.param_n
    idx = 0
    for key in state:
        state[key] = flat_params[idx].detach().cpu()
        idx += 1
    template_copy = copy.deepcopy(template)
    template_copy.load_state_dict(state)
    template_copy.eval()
    return template_copy


@torch.no_grad()
def evaluate_model(model, test_loader, loss_fn=None, device="cpu"):
    """Evaluate a model: returns (loss, accuracy) on the test set.

    Works for both classification (cross-entropy) and regression (MSE).
    """
    model = model.to(device)
    model.eval()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    is_classification = isinstance(loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss))

    for batch, labels in tqdm(test_loader, desc="Evaluating"):
        batch, labels = batch.to(device), labels.to(device)
        logits = model(batch)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * batch.size(0)
        total_samples += batch.size(0)

        if is_classification:
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples if is_classification else None
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# AGS Inference from Saved Bounds
# ---------------------------------------------------------------------------


def ags_inference_from_bounds(
    bounds_path: str,
    model: nn.Sequential,
    epsilon_total: float,
    *,
    device: str = "cpu",
    seed: int = 42,
    privacy_mode: Optional[str] = None,
    delta: Optional[float] = None,
    gamma_cauchy: int = 2,
    noise_mode: str = "anisotropic", # in {anisotropic, isotropic}
) -> nn.Sequential:
    """Re-evaluate AGS at a new epsilon using saved bounds (no retraining).

    Loads per-k upper/lower bounds saved by ``train_ags(..., save_bounds_path=...)``,
    recomputes smooth sensitivity for the given ``epsilon_total``, samples fresh
    noise, and returns the privatised model.

    Args:
        bounds_path: Path to the ``.pt`` file saved by ``train_ags``.
        model: A fresh nn.Sequential model with the same architecture.
        epsilon_total: Total privacy budget for the new evaluation.
        device: Device string.
        seed: Random seed for noise sampling.
        privacy_mode: Override the saved privacy mode (``"pure"`` or ``"approx"``).
            If None, uses the mode saved in the bounds file.
        delta: Override delta for approx DP. If None, uses saved value.

    Returns:
        model_out: nn.Sequential with privatised parameters for the new epsilon.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    bd = torch.load(bounds_path, map_location="cpu", weights_only=False)

    param_n = bd['param_n']
    per_k_bounds = bd['per_k_bounds']
    true_gs_window = bd['true_gs_window']
    k_subset = bd['k_subset']
    n_windows = bd['n_windows']
    saved_mode = bd['privacy_mode']

    if privacy_mode is None:
        privacy_mode = saved_mode
    if delta is None:
        delta = bd.get('delta', None)
    if privacy_mode == "approx" and delta is None:
        raise ValueError("delta must be provided for approx DP")

    # Move to device
    param_n = [p.to(device) for p in param_n]
    n_params = sum(p.numel() for p in param_n)
    true_gs_window = [g.to(device) for g in true_gs_window]

    # Epsilon per window (assuming single window)
    epsilon_per_window = epsilon_total / n_windows

    if privacy_mode == "pure":
        beta = epsilon_per_window / (2.0 * (gamma_cauchy + 1.0)) -1e-7 # epsilon / 6 when anisotropic and gamma_cauchy = 2
    else:  # approx
        # beta = epsilon_per_window / (2.0 * math.log(2.0 / delta)) - 1e-7
        beta = min(1.0, epsilon_per_window / (1.0 + 2 * np.e * float(np.log(n_params / delta))))

    # Reconstruct local_sens from saved per-k bounds
    local_sens = {}
    for k in k_subset:
        kb = per_k_bounds[k]
        pl = [p.to(device) for p in kb['param_l']]
        pu = [p.to(device) for p in kb['param_u']]
        widths = [(u - l).abs() for l, u in zip(pl, pu)]
        gs_k = [g * k for g in true_gs_window]
        local_sens[k] = [torch.minimum(w, g) for w, g in zip(widths, gs_k)]

    _log_local_sens_diagnostics(local_sens, true_gs_window, k_subset, "[inference] ")


    beta_js = [torch.full_like(p, beta / n_params) for p in param_n]
    if noise_mode == "anisotropic":
        # Compute growth rates from flattened local_sens, then split back to per-layer
        local_sens_flatten_per_k = {k: torch.cat([ls.flatten() for ls in local_sens[k]]) for k in k_subset}
        growth_rates_flat = torch.zeros(n_params, device=device)
        for k, km1 in zip(k_subset[1:], k_subset[:-1]):
            ls_k = local_sens_flatten_per_k[k].clamp(min=1e-30)
            ls_km1 = local_sens_flatten_per_k[km1].clamp(min=1e-30)
            growth_rates_flat += torch.log(ls_k / ls_km1)
        growth_rates_flat *= (1.0 / len(k_subset))
        # Split growth_rates back to per-layer and scale beta_js
        layer_numels = [p.numel() for p in param_n]
        gr_layers = list(growth_rates_flat.split(layer_numels))
        beta_js = [bj * gr.view(bj.shape) for bj, gr in zip(beta_js, gr_layers)]
        # Renormalise so that sum of all beta_j = beta
        beta_sum = sum(bj.sum().item() for bj in beta_js)
        beta_js = [beta * bj / beta_sum for bj in beta_js]

        logger.info(
            f"Anisotropic Growth Rates: mean={growth_rates_flat.mean().item():.4e}, "
            f"std={growth_rates_flat.std().item():.4e}, min={growth_rates_flat.min().item():.4e}, "
            f"max={growth_rates_flat.max().item():.4e}"
        )

    # Build staircase smooth sensitivity (same logic as train_ags)
    check_points = set()
    check_points.add(0)
    for kv in k_subset:
        check_points.add(kv)
    for j in range(len(k_subset) - 1):
        check_points.add(k_subset[j] + 1)
    check_points = sorted(check_points)

    max_k_val = k_subset[-1]

    def _get_staircase_ls(k_val):
        for kv in k_subset:
            if kv >= k_val:
                return local_sens[kv]
        return [gs * k_val for gs in true_gs_window]

    ss = [torch.zeros_like(p) for p in param_n]
    for cp in check_points:
        staircase_ls = _get_staircase_ls(cp)
        for i in range(len(ss)):
            decay = torch.exp(-beta_js[i] * cp)  # per-coordinate decay
            capped = torch.minimum(staircase_ls[i], true_gs_window[i])
            candidate = decay * capped
            ss[i] = torch.maximum(ss[i], candidate)

    # Tail check
    for i in range(len(ss)):
        tail_decay = torch.exp(-beta_js[i] * (max_k_val + 1))
        tail_candidate = tail_decay * true_gs_window[i]
        ss[i] = torch.maximum(ss[i], tail_candidate)

    if noise_mode == "anisotropic":
        ss_sqrt_sum = sum(s.sqrt().sum().item() for s in ss)
        a_js = [ss_sqrt_sum / s.sqrt().clamp(min=1e-30) for s in ss]
    else:
        a_js = [torch.full_like(s, float(n_params)) for s in ss]
    s_js = [s * a for s, a in zip(ss, a_js)]
    if privacy_mode == "approx":
        sigma_js = [(2.0 / epsilon_per_window) * sj for sj in s_js]
    else:  # pure
        sigma_js = [((2.0 * gamma_cauchy) / epsilon_per_window) * sj for sj in s_js]

    _log_multiplier_diagnostics(beta_js, ss, a_js, sigma_js, "[inference] ")

    # Sample noise and add to nominal parameters
    noisy_params = []
    for i, pn in enumerate(param_n):
        noise_scale = torch.clamp(sigma_js[i], min=1e-8)
        if privacy_mode == "approx":
            noise = torch.distributions.Laplace(
                torch.zeros_like(pn), noise_scale
            ).sample()
        else:
            noise = torch.distributions.Cauchy(
                torch.zeros_like(pn), noise_scale
            ).sample()
        noisy_params.append(pn + noise)

    # Load noisy params into model
    model = copy.deepcopy(model).to(device)
    idx = 0
    for p in model.parameters():
        p.data.copy_(noisy_params[idx])
        idx += 1
    model.eval()

    ss_flat = torch.cat([s.flatten() for s in ss])
    ss_mean = ss_flat.mean().item()
    ss_std = ss_flat.std().item()
    avg_noise_scale = sum(sj.mean().item() for sj in sigma_js) / len(sigma_js)
    logger.info(
        f"AGS inference from bounds: eps={epsilon_total:.2f}  "
        f"SS_mean={ss_mean:.2e} (std={ss_std:.2e})  avg_noise={avg_noise_scale:.2e}  mode={privacy_mode}"
    )

    return model


# ---------------------------------------------------------------------------
# Cross-radius single release (added alongside; train_ags / ags_inference_from_bounds above
# are the reference implementation and are unchanged)
# ---------------------------------------------------------------------------

def ags_release_cross_radius(
    bounds_path: Union[str, dict],
    model: nn.Sequential,
    epsilon_total: float,
    *,
    privacy_mode: Optional[str] = None,
    delta: Optional[float] = None,
    mechanism: Optional[str] = None,
    seed: int = 42,
    device: str = "cpu",
    return_diagnostics: bool = False,
):
    """Single AGS release calibrated to the paper's cross-radius certificate.

    Reads the per-k parameter envelopes saved by ``train_ags(..., save_bounds_path=...)`` and,
    per coordinate j, forms the sound-bounding function of Thm 4.2

        Abar_j(k) = max(|L_j^k - U_j^{k'}|, |U_j^k - L_j^{k'}|),   k' = next available radius,

    with k = 0 the nominal point, gaps back-filled by the next larger envelope, and the largest
    radius crossed with the clipping-ceiling box (nominal +/- sum_lr * gamma, i.e. every record
    substituted). The smooth sensitivity is SS_j = max_k e^{-beta k} Abar_j(k), and the release
    adds independent noise per coordinate with the budget split evenly across the P coordinates
    (basic composition, the isotropic ``a_j = P`` allocation of the reference implementation):

        pure   : Cauchy(6 SS_j / eps_c),   beta = eps_c / 6,               eps_c = eps / P
        approx : Laplace(2 SS_j / eps_c),  beta = eps_c / (2 ln(2/delta_c)), delta_c = delta / P

    Tighter coordinate accounting (PLD / RDP) lives in ``common.accounting``. This function is
    the certificate-convention counterpart of ``ags_inference_from_bounds``, which uses the
    envelope width with the closed-form cap.
    """
    from common.accounting.sensitivity import ReachableEnvelopes, cross_radius_a_bar, smooth_sensitivity

    torch.manual_seed(seed)
    np.random.seed(seed)
    bd = torch.load(bounds_path, map_location="cpu", weights_only=False) if isinstance(bounds_path, str) else bounds_path
    if privacy_mode is None:
        privacy_mode = bd["privacy_mode"]
    if delta is None:
        delta = bd.get("delta", None)
    if privacy_mode == "approx" and delta is None:
        raise ValueError("delta must be provided for approx DP")
    if mechanism is None:
        mechanism = "cauchy" if privacy_mode == "pure" else "laplace"
    if bd.get("n_windows", 1) != 1:
        raise ValueError("cross-radius release is defined for a single window (n_windows == 1)")

    flat = lambda ts: torch.cat([t.reshape(-1).double() for t in ts])
    theta = flat(bd["param_n"])
    P = theta.numel()
    k_subset = sorted(int(k) for k in bd["k_subset"])
    gs1 = flat(bd["true_gs_window"])                     # one-substitution GS per coordinate = sum_lr*2*gamma/b
    ceiling_half = gs1 * bd["batchsize"] / 2.0           # sum_lr * gamma: every record substituted
    kmax = k_subset[-1]
    L = torch.empty(kmax + 2, P, dtype=torch.float64); U = torch.empty_like(L)
    L[0] = theta; U[0] = theta
    nxt = None
    for k in range(kmax, 0, -1):
        if k in k_subset:
            kb = bd["per_k_bounds"][k]
            nxt = (torch.minimum(flat(kb["param_l"]), theta), torch.maximum(flat(kb["param_u"]), theta))
        L[k], U[k] = nxt
    L[kmax + 1] = torch.minimum(theta - ceiling_half, L[kmax]); U[kmax + 1] = torch.maximum(theta + ceiling_half, U[kmax])
    for k in range(1, kmax + 2):                        # enforce nesting (running hull)
        L[k] = torch.minimum(L[k], L[k - 1]); U[k] = torch.maximum(U[k], U[k - 1])
    a_bar = cross_radius_a_bar(ReachableEnvelopes(L, U))   # rows 0..kmax+1; last row crosses ceiling with itself

    eps_c = epsilon_total / P
    if mechanism == "cauchy":
        beta = eps_c / 6.0 - 1e-12
    elif mechanism == "laplace":
        delta_c = delta / P
        beta = eps_c / (2.0 * math.log(2.0 / delta_c))
    else:
        raise ValueError(mechanism)
    ss = smooth_sensitivity(a_bar, beta)                     # per coordinate
    if mechanism == "cauchy":
        scale = 6.0 * ss / eps_c
        noise = torch.distributions.Cauchy(torch.zeros(P, dtype=torch.float64), scale.clamp_min(1e-300)).sample()
    else:
        scale = 2.0 * ss / eps_c
        noise = torch.distributions.Laplace(torch.zeros(P, dtype=torch.float64), scale.clamp_min(1e-300)).sample()
    theta_priv = theta + noise

    out = copy.deepcopy(model).to(device)
    i = 0
    with torch.no_grad():
        for prm in out.parameters():
            n = prm.numel()
            prm.copy_(theta_priv[i:i + n].reshape(prm.shape).to(prm.dtype).to(device)); i += n
    assert i == P
    if return_diagnostics:
        return out, dict(a_bar=a_bar, smooth_sensitivity=ss, scale=scale, beta=beta, eps_coord=eps_c, P=P,
                         mechanism=mechanism, k_subset=k_subset)
    return out
