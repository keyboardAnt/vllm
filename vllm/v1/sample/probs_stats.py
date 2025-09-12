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


_GLOBAL_PROBS_STATS: OnlineMeanStd | None = None


def _get_global() -> OnlineMeanStd:
    global _GLOBAL_PROBS_STATS
    if _GLOBAL_PROBS_STATS is None:
        _GLOBAL_PROBS_STATS = OnlineMeanStd()
    return _GLOBAL_PROBS_STATS


@torch.no_grad()
def update_global_probs_stats(x: torch.Tensor) -> None:
    """Update the global per-token-id statistics accumulator.

    This singleton accumulator aggregates across the entire process lifetime
    and can be queried at teardown.
    """
    _get_global().update(x)


@torch.no_grad()
def get_global_probs_stats() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the global (mean, std) per-token-id statistics."""
    return _get_global().get()


@torch.no_grad()
def reset_global_probs_stats() -> None:
    """Reset the global accumulator."""
    _get_global().reset()


def visualize_per_token_stats(mean: torch.Tensor,
                              std: torch.Tensor | None,
                              output_path: str,
                              top_k: int = 50) -> str:
    """Visualize per-token-id mean (and optional std) statistics.

    This function is intended to be called once at the end of a benchmark.
    If matplotlib is available, it saves a figure; otherwise it falls back
    to saving a CSV file containing the top-K tokens by mean (and std if
    provided).

    Args:
        mean: 1D tensor of shape [vocab_size], mean per token id.
        std: Optional 1D tensor of shape [vocab_size], std per token id.
        output_path: Path to save the visualization (e.g., "probs_stats.png").
        top_k: Number of top tokens to show in the bar chart.

    Returns:
        The path of the created file (PNG or CSV).
    """
    if mean.dim() != 1:
        raise ValueError("visualize_per_token_stats expects mean of shape [vocab_size]")
    if std is not None and std.dim() != 1:
        raise ValueError("visualize_per_token_stats expects std of shape [vocab_size]")

    # Move to CPU float64 for stable plotting/saving.
    mean_cpu = mean.detach().to(dtype=torch.float64, device="cpu")
    std_cpu = None if std is None else std.detach().to(dtype=torch.float64, device="cpu")
    vocab_size = int(mean_cpu.numel())
    top_k = int(max(1, min(int(top_k), vocab_size)))

    # Try to plot with matplotlib; fall back to CSV if unavailable.
    try:
        import os
        import importlib
        # Dynamically import matplotlib only if available to avoid linter/env issues
        if importlib.util.find_spec("matplotlib") is None:
            raise ImportError("matplotlib not available")
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg", force=True)
        plt = importlib.import_module("matplotlib.pyplot")

        # Compute top-k by mean (keep tensors for indexing, then convert)
        top_vals_t, top_idx_t = torch.topk(mean_cpu, k=top_k, largest=True)
        order_t = torch.argsort(top_vals_t)  # ascending for nicer bar order
        top_vals_np = top_vals_t[order_t].numpy()
        top_idx_t = top_idx_t[order_t]
        top_idx_np = top_idx_t.numpy()

        if std_cpu is not None:
            top_std_np = std_cpu[top_idx_t].numpy()

        # Figure layout: if std provided, use 2x2; else use 1x2
        if std_cpu is not None:
            fig = plt.figure(figsize=(14, 10))
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            ax1.hist(mean_cpu.numpy(), bins=50, color="#4e79a7")
            ax1.set_title("Per-token mean distribution")
            ax1.set_xlabel("mean")
            ax1.set_ylabel("count")

            ax2.hist(std_cpu.numpy(), bins=50, color="#59a14f")
            ax2.set_title("Per-token std distribution")
            ax2.set_xlabel("std")
            ax2.set_ylabel("count")

            ax3.bar(range(top_k), top_vals_np, color="#f28e2b")
            ax3.set_title(f"Top-{top_k} token means")
            ax3.set_xlabel("token rank (ascending)")
            ax3.set_ylabel("mean")
            ax3.set_xticks(range(0, top_k, max(1, top_k // 10)))
            sparse_labels = [str(int(i)) for i in top_idx_np[::max(1, top_k // 10)]]
            ax3.set_xticklabels(sparse_labels, rotation=45, ha="right")

            ax4.bar(range(top_k), top_std_np, color="#e15759")
            ax4.set_title(f"Top-{top_k} token std (by mean's top-K order)")
            ax4.set_xlabel("token rank (ascending)")
            ax4.set_ylabel("std")
            ax4.set_xticks(range(0, top_k, max(1, top_k // 10)))
            ax4.set_xticklabels(sparse_labels, rotation=45, ha="right")
        else:
            fig = plt.figure(figsize=(12, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.hist(mean_cpu.numpy(), bins=50, color="#4e79a7")
            ax1.set_title("Per-token mean distribution")
            ax1.set_xlabel("mean")
            ax1.set_ylabel("count")

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.bar(range(top_k), top_vals_np, color="#f28e2b")
            ax2.set_title(f"Top-{top_k} token means")
            ax2.set_xlabel("token rank (ascending)")
            ax2.set_ylabel("mean")
            ax2.set_xticks(range(0, top_k, max(1, top_k // 10)))
            sparse_labels = [str(int(i)) for i in top_idx_np[::max(1, top_k // 10)]]
            ax2.set_xticklabels(sparse_labels, rotation=45, ha="right")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    except Exception:
        # Fallback: save CSV with top-K means (and std if available)
        import os
        import numpy as np
        csv_path = output_path
        if csv_path.lower().endswith((".png", ".jpg", ".jpeg")):
            csv_path = os.path.splitext(csv_path)[0] + ".csv"
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        top_vals, top_idx = torch.topk(mean_cpu, k=top_k, largest=True)
        top_idx_np = top_idx.numpy()
        top_vals_np = top_vals.numpy()
        if std_cpu is not None:
            top_std_np = std_cpu[top_idx].numpy()
            data = np.stack([top_idx_np, top_vals_np, top_std_np], axis=1)
            header = "token_id,mean,std"
        else:
            data = np.stack([top_idx_np, top_vals_np], axis=1)
            header = "token_id,mean"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt=["%d", "%.10f"] + (["%.10f"] if std_cpu is not None else []))
        return csv_path

