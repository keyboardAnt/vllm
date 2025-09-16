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

import os
import glob
import torch
from vllm.logger import init_logger


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


logger = init_logger(__name__)

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
    stats = _get_global()

    # Log stats of x
    logger.info(f"{x.shape=}")
    logger.info(f"{x.mean(dim=-1).mean()=}")
    logger.info(f"{x.std(dim=-1).mean()=}")
    logger.info(f"{x.min()=}")
    assert x.min() >= 0.0
    logger.info(f"{x.max()=}")
    assert x.max() <= 1.0
    assert torch.allclose(x.sum(dim=-1), 1.0)

    stats.update(x)
    # Log a brief summary of current global stats.
    try:
        mean, std = stats.get()
        logger.info(
            "Target probs stats (global): count=%d, dim=%d, mean_mean=%.6f, std_mean=%.6f",
            stats.count,
            mean.numel(),
            float(mean.mean()),
            float(std.mean()),
        )
    except Exception as e:  # noqa: BLE001
        # No observations yet or intermediate state; skip quietly.
        logger.debug(f"Global OnlineMeanStd get skipped: {e}")

    # Persist the current accumulators to a per-process file. If the output
    # directory is not provided via env, default to "probs_stats" in CWD.
    stats_dir = os.environ.get("VLLM_PROBS_STATS_DIR", "probs_stats")
    try:
        os.makedirs(stats_dir, exist_ok=True)
        file_path = os.path.join(stats_dir, f"probs_stats_{os.getpid()}.pt")
        # Save CPU float64 for numerical stability when aggregating.
        payload = {
            "count": stats.count,
            "mean": stats.mean.detach().to(dtype=torch.float64, device="cpu")
            if stats.mean is not None else None,
            "M2": stats.M2.detach().to(dtype=torch.float64, device="cpu")
            if stats.M2 is not None else None,
        }
        torch.save(payload, file_path)
    except Exception as save_e:  # noqa: BLE001
        logger.debug(f"Failed to persist global probs stats: {save_e}")
    


@torch.no_grad()
def get_global_probs_stats() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the global (mean, std) per-token-id statistics."""
    return _get_global().get()


@torch.no_grad()
def reset_global_probs_stats() -> None:
    """Reset the global accumulator."""
    _get_global().reset()


@torch.no_grad()
def aggregate_saved_probs_stats(stats_dir: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Aggregate per-process saved stats into global mean/std.

    The directory should contain files saved by update_global_probs_stats with
    names like "probs_stats_<pid>.pt" that include keys: count, mean, M2.

    Returns:
        (mean, std, count): Aggregated tensors (float64 CPU) and total count.
    """
    files = sorted(glob.glob(os.path.join(stats_dir, "probs_stats_*.pt")))
    if not files:
        raise ValueError(f"No stats files found in {stats_dir}")

    total_count = 0
    agg_mean = None
    agg_M2 = None

    for fp in files:
        try:
            data = torch.load(fp, map_location="cpu")
            cnt = int(data.get("count", 0))
            mean = data.get("mean", None)
            M2 = data.get("M2", None)
            if cnt <= 0 or mean is None or M2 is None:
                continue
            mean = mean.to(dtype=torch.float64, device="cpu")
            M2 = M2.to(dtype=torch.float64, device="cpu")
        except Exception:
            continue

        if total_count == 0:
            agg_mean = mean.clone()
            agg_M2 = M2.clone()
            total_count = cnt
        else:
            assert agg_mean is not None and agg_M2 is not None
            delta = mean - agg_mean
            new_total = total_count + cnt
            agg_mean = agg_mean + delta * (cnt / new_total)
            agg_M2 = agg_M2 + M2 + delta * delta * (total_count * cnt / new_total)
            total_count = new_total

    if agg_mean is None or agg_M2 is None or total_count == 0:
        raise ValueError(f"No valid stats found in {stats_dir}")

    if total_count > 1:
        var = agg_M2 / (total_count - 1)
    else:
        var = torch.zeros_like(agg_M2)
    std = torch.sqrt(torch.clamp(var, min=0))
    return agg_mean, std, total_count


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
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        # Sort tokens by descending mean probability and select top_k
        order_t = torch.argsort(mean_cpu, descending=True)
        sel_t = order_t[:top_k]
        mean_sorted = mean_cpu[sel_t].numpy()

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        x = np.arange(mean_sorted.shape[0])
        if std_cpu is not None:
            std_sorted = std_cpu[sel_t].numpy()
            # Clip the vertical span to be non-negative
            lower = np.maximum(0.0, mean_sorted - std_sorted)
            upper = mean_sorted + std_sorted
            ax.vlines(
                x,
                lower,
                upper,
                color="orange",
                alpha=0.6,
                linewidth=0.5,
                label="±1 std (lower clipped at 0)",
            )
            ax.plot(x, mean_sorted, color="navy", linewidth=1.2, label="Mean probability")
            ax.set_title("Sorted per-token probabilities with ±1 std deviation bars")
            ax.legend()
        else:
            ax.plot(x, mean_sorted, color="navy", linewidth=1.2, label="Mean probability")
            ax.set_title("Sorted per-token probabilities")

        ax.set_xlabel("Token (sorted by mean probability)")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
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
        # Sort and take top_k to mirror the visualization
        order_t = torch.argsort(mean_cpu, descending=True)
        sel_t = order_t[:top_k]
        top_idx_np = sel_t.numpy()
        top_vals_np = mean_cpu[sel_t].numpy()
        if std_cpu is not None:
            top_std_np = std_cpu[sel_t].numpy()
            data = np.stack([top_idx_np, top_vals_np, top_std_np], axis=1)
            header = "token_id,mean,std"
        else:
            data = np.stack([top_idx_np, top_vals_np], axis=1)
            header = "token_id,mean"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt=["%d", "%.10f"] + (["%.10f"] if std_cpu is not None else []))
        return csv_path

