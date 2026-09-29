"""The training arms the accounting experiments compare.

Four arms over the same data, model and schedule, so a difference between them is the mechanism and
nothing else: non-private, clipped but noiseless, AGS (one release of the parameter vector), and
DP-SGD (noise at every step).

Noise is drawn through an inverse CDF from uniforms, which is what makes a release reproducible from
a seed and was also the fix for a float32 endpoint defect that corrupted one screen. `train_ags`
builds the reachable envelopes, takes the cross-radius certificate from `sensitivity.py`, and scales
the release per coordinate.
"""

from __future__ import annotations

import math
import resource
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters

from .accounting import Calibration
from .config import ExperimentConfig
from .gradients import coordinate_clip
from .gradients import l2_clip
from .gradients import per_example_gradients
from .model import layer_slices
from .sensitivity import coordinate_global_sensitivity
from .sensitivity import clipped_mean_smooth_sensitivity
from .sensitivity import cross_radius_a_bar
from .sensitivity import one_step_envelopes
from .sensitivity import smooth_sensitivity


class _Progress:
    """Epoch-level progress reporting without releasing model state."""

    def __init__(self, arm: str, steps: int, steps_per_epoch: int) -> None:
        self.arm = arm
        self.steps = steps
        self.steps_per_epoch = steps_per_epoch
        self.report_every = max(1, steps // 10)
        self.started = time.perf_counter()

    def update(self, completed: int) -> None:
        if completed % self.report_every and completed != self.steps:
            return
        elapsed = time.perf_counter() - self.started
        eta = elapsed / completed * (self.steps - completed)
        epoch = completed / self.steps_per_epoch
        print(
            f"[{self.arm}] epoch_progress={epoch:.2f} "
            f"step={completed}/{self.steps} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _noise(shape: torch.Size, scale: torch.Tensor, family: str, generator: torch.Generator) -> torch.Tensor:
    uniform = torch.rand(shape, generator=generator, dtype=scale.dtype)
    uniform = uniform.to(scale.device)
    return _noise_from_uniform(uniform, scale, family)


def _noise_from_uniform(uniform: torch.Tensor, scale: torch.Tensor, family: str) -> torch.Tensor:
    """Inverse-CDF noise with endpoints clamped in the actual tensor dtype."""
    epsilon = torch.finfo(uniform.dtype).eps
    uniform = uniform.clamp(epsilon, 1 - epsilon)
    if family == "cauchy":
        return scale * torch.tan(math.pi * (uniform - 0.5))
    if family == "laplace":
        centered = uniform - 0.5
        return -scale * centered.sign() * torch.log1p(-2 * centered.abs())
    raise ValueError(f"unknown noise family {family}")


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            correct += int((logits.argmax(1) == targets).sum())
            count += len(inputs)
    return correct / count, total_loss / count


def _diagnostics(
    model: nn.Module, sensitivity: torch.Tensor, gs: torch.Tensor, scale: torch.Tensor
) -> dict[str, dict[str, list[float]]]:
    gs = gs.to(sensitivity.device)
    scale = scale.to(sensitivity.device)
    output = {}
    for name, section in layer_slices(model).items():
        ratio = sensitivity[section] / gs[section].clamp_min(torch.finfo(gs.dtype).tiny)
        values = scale[section]
        quantiles = torch.tensor([0, 0.25, 0.5, 0.75, 1.0], dtype=ratio.dtype, device=ratio.device)
        output[name] = {
            "s_over_gs_quantiles": torch.quantile(ratio, quantiles).tolist(),
            "noise_scale_quantiles": torch.quantile(values, quantiles).tolist(),
        }
    return output


def _direct_diagnostics(
    model: nn.Module,
    clipped: torch.Tensor,
    caps: torch.Tensor,
    sensitivity: torch.Tensor,
    gs: torch.Tensor,
    scale: torch.Tensor,
    beta: float,
    learning_rate: float,
) -> dict[str, object]:
    """Explain which term determines each direct smooth-sensitivity scale."""
    caps = caps.to(clipped.device)
    gs = gs.to(clipped.device)
    sensitivity = sensitivity.to(clipped.device)
    scale = scale.to(clipped.device)
    max_abs = clipped.abs().amax(dim=0)
    local = learning_rate * (caps + max_abs) / clipped.shape[0]
    beta_floor = math.exp(-beta) * gs
    signal = learning_rate * clipped.mean(dim=0).abs()
    result: dict[str, object] = _diagnostics(model, sensitivity, gs, scale)
    for name, section in layer_slices(model).items():
        dtype = clipped.dtype
        quantiles = torch.tensor([0, 0.25, 0.5, 0.75, 1.0], dtype=dtype, device=clipped.device)
        result[name].update(
            {
                "local_over_gs_quantiles": torch.quantile(
                    local[section] / gs[section].clamp_min(torch.finfo(dtype).tiny), quantiles
                ).tolist(),
                "signal_over_noise_quantiles": torch.quantile(
                    signal[section] / scale[section].clamp_min(torch.finfo(dtype).tiny), quantiles
                ).tolist(),
                "cap_reached_fraction": float(
                    torch.isclose(max_abs[section], caps[section], rtol=1e-5, atol=1e-8)
                    .to(torch.float64)
                    .mean()
                ),
                "local_term_selected_fraction": float(
                    (local[section] >= beta_floor[section]).to(torch.float64).mean()
                ),
            }
        )
    result["certificate"] = {
        "method": "direct_clipped_mean",
        "beta": beta,
        "exp_minus_beta": math.exp(-beta),
    }
    return result


def public_coordinate_caps(model: nn.Module, config: ExperimentConfig) -> torch.Tensor:
    """Construct immutable per-coordinate caps from public parameter roles."""
    caps = torch.full(
        (sum(parameter.numel() for parameter in model.parameters()),),
        config.coordinate_clip,
        device="cpu",
    )
    for name, section in layer_slices(model).items():
        if name.endswith(".weight") and config.coordinate_clip_weight is not None:
            caps[section] = config.coordinate_clip_weight
        elif name.endswith(".bias") and config.coordinate_clip_bias is not None:
            caps[section] = config.coordinate_clip_bias
    return caps


def ags_release_count(model: nn.Module, mode: str) -> int:
    """Number of mechanisms composed before the single sampling amplification."""
    if mode == "coordinate":
        return sum(parameter.numel() for parameter in model.parameters())
    if mode == "tensor_l1":
        return len(tuple(model.parameters()))
    if mode == "joint_l1":
        return 1
    raise ValueError(f"unsupported AGS release mode {mode}")


def ags_release_scale(
    model: nn.Module,
    coordinate_sensitivity: torch.Tensor,
    epsilon: float,
    mode: str,
    family: str,
) -> torch.Tensor:
    """Map coordinate bounds to sound coordinate or l1-block noise scales."""
    multiplier = 1.0 if family == "cauchy" else 2.0
    if mode == "coordinate":
        return multiplier * coordinate_sensitivity / epsilon
    scale = torch.empty_like(coordinate_sensitivity)
    if mode == "joint_l1":
        scale.fill_(multiplier * float(coordinate_sensitivity.sum()) / epsilon)
        return scale
    if mode == "tensor_l1":
        for section in layer_slices(model).values():
            scale[section] = multiplier * coordinate_sensitivity[section].sum() / epsilon
        return scale
    raise ValueError(f"unsupported AGS release mode {mode}")


def train_nonprivate(
    model: nn.Module, train_set, test_loader, config: ExperimentConfig, device: torch.device
) -> dict[str, object]:
    started = time.perf_counter()
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    generator = _generator(config.seed + 101)
    steps_per_epoch = math.ceil(len(train_set) / config.batch_size)
    steps = steps_per_epoch * config.epochs
    progress = _Progress("nonprivate", steps, steps_per_epoch)
    for step_index in range(steps):
        indices = torch.randperm(len(train_set), generator=generator)[: config.batch_size]
        inputs, targets = _indexed_batch(train_set, indices, device)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(inputs), targets).backward()
        optimizer.step()
        progress.update(step_index + 1)
    accuracy, loss = evaluate(model, test_loader, device)
    return _result("nonprivate", accuracy, loss, started, steps, None, {})


def train_clipped_noiseless(
    model: nn.Module,
    train_set,
    evaluation_loader,
    config: ExperimentConfig,
    device: torch.device,
    *,
    geometry: str,
) -> dict[str, object]:
    """Diagnostic SGD with per-example clipping and no privacy noise."""
    if geometry not in {"coordinate", "l2"}:
        raise ValueError("geometry must be coordinate or l2")
    started = time.perf_counter()
    model.to(device)
    sample_generator = _generator(config.seed + 601)
    steps_per_epoch = math.ceil(len(train_set) / config.batch_size)
    steps = steps_per_epoch * config.epochs
    progress = _Progress(f"clipped_{geometry}", steps, steps_per_epoch)
    clipping_rates = []
    for step_index in range(steps):
        indices = torch.randperm(len(train_set), generator=sample_generator)[: config.batch_size]
        inputs, targets = _indexed_batch(train_set, indices, device)
        grads = per_example_gradients(
            model,
            inputs,
            targets,
            config.gradient_chunk_size,
            output_device=device,
        )
        if geometry == "coordinate":
            caps = public_coordinate_caps(model, config).to(device)
            clipped, rate = coordinate_clip(grads, caps)
        else:
            clipped, rate = l2_clip(grads, config.l2_clip)
        updated = parameters_to_vector(model.parameters()).detach() - (config.learning_rate * clipped.mean(0))
        vector_to_parameters(updated, model.parameters())
        clipping_rates.append(rate)
        progress.update(step_index + 1)
    accuracy, loss = evaluate(model, evaluation_loader, device)
    return _result(
        f"clipped_{geometry}",
        accuracy,
        loss,
        started,
        steps,
        None,
        {"clipping_rate": float(np.mean(clipping_rates))},
    )


def train_ags(
    model: nn.Module,
    train_set,
    test_loader,
    config: ExperimentConfig,
    calibration: Calibration,
    device: torch.device,
    *,
    use_gs: bool,
) -> dict[str, object]:
    if not use_gs and config.ags_release_mode != "coordinate":
        raise ValueError(
            "data-dependent tensor/joint Laplace scales require a dimension-aware "
            "smoothness theorem; use coordinate mode or the fixed-scale GS ablation"
        )
    started = time.perf_counter()
    model.to(device)
    p = sum(x.numel() for x in model.parameters())
    caps = public_coordinate_caps(model, config)
    gs = coordinate_global_sensitivity(caps, config.batch_size, config.learning_rate)
    sample_generator = _generator(config.seed + 201)
    noise_generator = _generator(config.seed + 301)
    steps_per_epoch = math.ceil(len(train_set) / config.batch_size)
    steps = steps_per_epoch * config.epochs
    progress = _Progress("ags_gs" if use_gs else "ags_ss", steps, steps_per_epoch)
    clipping_rates = []
    last_diagnostics = {}
    for step_index in range(steps):
        indices = torch.randperm(len(train_set), generator=sample_generator)[: config.batch_size]
        inputs, targets = _indexed_batch(train_set, indices, device)
        matrix_bytes = config.batch_size * p * 4
        gradient_device = torch.device("cpu") if matrix_bytes > 512 * 2**20 else device
        grads = per_example_gradients(
            model,
            inputs,
            targets,
            config.gradient_chunk_size,
            output_device=gradient_device,
            progress_label=f"ags_ss:step{step_index + 1}",
        )
        local_caps = caps.to(gradient_device)
        clipped, rate = coordinate_clip(grads, local_caps)
        clipping_rates.append(rate)
        clean_current = parameters_to_vector(model.parameters()).detach().to(gradient_device)
        nominal = clean_current - config.learning_rate * clipped.mean(0)
        if use_gs:
            sensitivity = gs
            a_bar = None
        else:
            beta = calibration.mechanism_parameter * config.beta_epsilon_ratio
            if config.sensitivity_method == "direct_clipped_mean":
                sensitivity = clipped_mean_smooth_sensitivity(
                    clipped, local_caps, beta, learning_rate=config.learning_rate
                )
                a_bar = None
            else:
                envelopes = one_step_envelopes(
                    clipped,
                    local_caps,
                    learning_rate=config.learning_rate,
                    theta=clean_current,
                )
                a_bar = cross_radius_a_bar(envelopes)
                sensitivity = smooth_sensitivity(a_bar, beta)
        eps_coord = calibration.mechanism_parameter
        beta = eps_coord * config.beta_epsilon_ratio
        release_count = ags_release_count(model, config.ags_release_mode)
        if config.ags_mechanism == "cauchy":
            # The Cauchy theorem uses epsilon-beta rather than epsilon.
            scale = ags_release_scale(
                model,
                sensitivity,
                eps_coord - beta,
                config.ags_release_mode,
                "cauchy",
            )
        else:
            delta_coord = (
                config.target_delta
                * config.ags_delta_fraction
                / (calibration.q * calibration.steps * 2 * release_count)
            )
            beta_limit = eps_coord / (2 * math.log(2 / delta_coord))
            if not use_gs and beta > beta_limit:
                raise ValueError("configured beta violates smooth-Laplace condition")
            scale = ags_release_scale(model, sensitivity, eps_coord, config.ags_release_mode, "laplace")
        released = nominal + _noise(nominal.shape, scale, config.ags_mechanism, noise_generator).to(
            nominal.device
        )
        # Privacy boundary: the sole persistent state is overwritten by the release.
        vector_to_parameters(released.to(device), model.parameters())
        del clean_current, nominal, grads, released
        if a_bar is not None and step_index == steps - 1:
            next_width = envelopes.upper[1:] - envelopes.lower[1:]
            final_width = envelopes.upper[-1:] - envelopes.lower[-1:]
            widths = torch.cat([next_width, final_width])
            last_diagnostics = _diagnostics(model, sensitivity, gs, scale)
            last_diagnostics["cross_radius_over_next_width"] = {
                "quantiles": torch.quantile(
                    (a_bar / widths.clamp_min(torch.finfo(widths.dtype).tiny)).reshape(-1),
                    torch.tensor([0, 0.25, 0.5, 0.75, 1.0], device=device, dtype=widths.dtype),
                ).tolist()
            }
        elif not use_gs and step_index == steps - 1:
            last_diagnostics = _direct_diagnostics(
                model,
                clipped,
                local_caps,
                sensitivity,
                gs,
                scale,
                beta,
                config.learning_rate,
            )
        del clipped
        progress.update(step_index + 1)
    accuracy, loss = evaluate(model, test_loader, device)
    extra = {"coordinate_clipping_rate": float(np.mean(clipping_rates)), "layers": last_diagnostics}
    return _result("ags_gs" if use_gs else "ags_ss", accuracy, loss, started, steps, calibration, extra)


def train_dpsgd(
    model: nn.Module,
    train_set,
    test_loader,
    config: ExperimentConfig,
    calibration: Calibration,
    device: torch.device,
) -> dict[str, object]:
    started = time.perf_counter()
    model.to(device)
    sample_generator = _generator(config.seed + 401)
    noise_generator = _generator(config.seed + 501)
    q = config.batch_size / len(train_set)
    steps_per_epoch = math.ceil(len(train_set) / config.batch_size)
    steps = steps_per_epoch * config.epochs
    progress = _Progress("dpsgd", steps, steps_per_epoch)
    clipping_rates = []
    for step_index in range(steps):
        mask = torch.rand(len(train_set), generator=sample_generator) < q
        indices = mask.nonzero().reshape(-1)
        if len(indices):
            inputs, targets = _indexed_batch(train_set, indices, device)
            p = sum(x.numel() for x in model.parameters())
            matrix_bytes = len(indices) * p * 4
            gradient_device = torch.device("cpu") if matrix_bytes > 512 * 2**20 else device
            grads = per_example_gradients(
                model,
                inputs,
                targets,
                config.gradient_chunk_size,
                output_device=gradient_device,
                progress_label=f"dpsgd:step{step_index + 1}",
            )
            clipped, rate = l2_clip(grads, config.l2_clip)
            gradient_sum = clipped.sum(0)
            clipping_rates.append(rate)
        else:
            gradient_sum = torch.zeros(sum(x.numel() for x in model.parameters()), device=device)
            clipping_rates.append(0.0)
        gaussian = torch.randn(gradient_sum.shape, generator=noise_generator, dtype=gradient_sum.dtype).to(
            gradient_sum.device
        )
        gradient_sum += gaussian * calibration.mechanism_parameter * config.l2_clip
        updated = parameters_to_vector(model.parameters()).detach() - (
            config.learning_rate * gradient_sum.to(device) / (q * len(train_set))
        )
        vector_to_parameters(updated, model.parameters())
        progress.update(step_index + 1)
    accuracy, loss = evaluate(model, test_loader, device)
    return _result(
        "dpsgd",
        accuracy,
        loss,
        started,
        steps,
        calibration,
        {"example_l2_clipping_rate": float(np.mean(clipping_rates))},
    )


def _result(
    arm: str,
    accuracy: float,
    loss: float,
    started: float,
    steps: int,
    calibration: Calibration | None,
    extra: dict[str, object],
) -> dict[str, object]:
    result = {
        "arm": arm,
        "test_accuracy": accuracy,
        "test_loss": loss,
        "runtime_seconds": time.perf_counter() - started,
        "steps": steps,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "privacy": None if calibration is None else asdict(calibration),
    }
    result.update(extra)
    return result


def _indexed_batch(
    train_set, indices: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index a preloaded TensorDataset without repeated PIL transformations."""
    if hasattr(train_set, "tensors"):
        inputs, targets = train_set.tensors
        return inputs[indices].to(device), targets[indices].to(device)
    return (
        torch.stack([train_set[int(i)][0] for i in indices]).to(device),
        torch.tensor([train_set[int(i)][1] for i in indices], device=device),
    )
