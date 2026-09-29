"""Brute-force ground truth on a finite domain, for testing the certificate.

Enumerates every dataset within a substitution radius and computes the reachable set and the local
sensitivity exactly. Far too slow for anything real, which is the point: it is the oracle the
cross-radius implementation in `sensitivity.py` is checked against in
`common/accounting/tests/test_sensitivity.py`.

A certificate that is merely self-consistent can still be wrong. This is what makes it falsifiable.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import torch


Dataset = tuple[int, ...]


def substitution_distance(left: Dataset, right: Dataset) -> int:
    if len(left) != len(right):
        raise ValueError("fixed-size substitution distance needs equal lengths")
    return sum(x != y for x, y in zip(left, right, strict=True))


def all_datasets(domain: tuple[int, ...], m: int) -> tuple[Dataset, ...]:
    return tuple(itertools.product(domain, repeat=m))


def exact_a(
    nominal: Dataset,
    datasets: tuple[Dataset, ...],
    output: Callable[[Dataset], torch.Tensor],
) -> torch.Tensor:
    """Enumerate the literal coordinate Nissim ``A_j^(k)`` definition."""
    m = len(nominal)
    sample = output(nominal)
    result = torch.zeros((m + 1, sample.numel()), dtype=sample.dtype)
    values = {dataset: output(dataset).reshape(-1) for dataset in datasets}
    for k in range(m + 1):
        candidates = []
        for dataset in datasets:
            if substitution_distance(nominal, dataset) > k:
                continue
            for neighbor in datasets:
                if substitution_distance(dataset, neighbor) == 1:
                    candidates.append((values[dataset] - values[neighbor]).abs())
        result[k] = torch.stack(candidates).max(dim=0).values
    return result


def exact_envelopes(
    nominal: Dataset,
    datasets: tuple[Dataset, ...],
    output: Callable[[Dataset], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate exact coordinate boxes ``T_k(D)``."""
    m = len(nominal)
    lower = []
    upper = []
    for k in range(m + 1):
        values = torch.stack(
            [
                output(dataset).reshape(-1)
                for dataset in datasets
                if substitution_distance(nominal, dataset) <= k
            ]
        )
        lower.append(values.min(dim=0).values)
        upper.append(values.max(dim=0).values)
    return torch.stack(lower), torch.stack(upper)
