"""Utilities for online statistics over probability vectors.

Implements numerically stable online mean and standard deviation using
Welford's algorithm, with support for batched updates.

Notes
-----
- The feature dimension corresponds to vocabulary token ids (per-token-id
  statistics). When inputs have shape [N, D], D should be the vocabulary
  size, and each column aggregates statistics for a specific token id across
  all observations (rows).
"""

from __future__ import annotations

from typing import Tuple

import torch


class OnlineMeanStd:
    """Tracks mean and standard deviation online for vector observations.

    The implementation uses the parallel form of Welford's algorithm and
    supports both single observation tensors of shape [D] and batched
    tensors of shape [N, D]. Internal accumulators use float64 for better
    numerical stability.

    Per-token-id semantics
    ----------------------
    Each of the D features corresponds to a vocabulary token id. When used
    with model outputs (e.g., `target_probs` of shape [num_tokens, vocab_size]),
    each row is one observation, and statistics are aggregated per token id
    (i.e., column-wise across rows).
    """

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: torch.Tensor | None = None
        self.M2: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with new observation(s).

        Args:
            x: A tensor of shape [D] or [N, D]. The first dimension, if
               present, is treated as the batch of independent observations.
               The feature dimension D corresponds to vocabulary token ids
               (i.e., usually the `vocab_size`).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError("OnlineMeanStd.update expects [D] or [N, D] input")

        x64 = x.detach().to(dtype=torch.float64)

        if self.mean is None:
            device = x64.device
            feature_dim = x64.size(1)
            self.mean = torch.zeros(feature_dim, dtype=torch.float64, device=device)
            self.M2 = torch.zeros(feature_dim, dtype=torch.float64, device=device)

        assert self.mean is not None and self.M2 is not None

        batch_count = x64.size(0)
        batch_mean = x64.mean(dim=0)
        # Sum of squared deviations within the batch
        batch_M2 = ((x64 - batch_mean) * (x64 - batch_mean)).sum(dim=0)

        total_count = self.count + batch_count
        if total_count == 0:
            return

        delta = batch_mean - self.mean
        # Update mean and M2 using the parallel update formula
        self.mean = self.mean + delta * (batch_count / total_count)
        self.M2 = self.M2 + batch_M2 + delta * delta * (self.count * batch_count / total_count)
        self.count = total_count

    @torch.no_grad()
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current mean and standard deviation tensors.

        Returns:
            (mean, std): Tensors of shape [D], where D is the number of
            vocabulary token ids (features). If fewer than two observations
            have been seen, the standard deviation is zeros.
        """
        if self.mean is None or self.M2 is None or self.count == 0:
            raise ValueError("No observations seen.")
        if self.count > 1:
            var = self.M2 / (self.count - 1)
        else:
            var = torch.zeros_like(self.M2)
        std = torch.sqrt(torch.clamp(var, min=0))
        return self.mean, std

    @torch.no_grad()
    def reset(self) -> None:
        """Reset the accumulators."""
        self.count = 0
        self.mean = None
        self.M2 = None


