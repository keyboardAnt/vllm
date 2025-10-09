"""Utilities for online statistics over vector features.

Implements numerically stable online mean and standard deviation using
Welford's algorithm, with support for batched updates.

Typical LLM use-cases tracked by this module include per-token-id statistics
for:
- target probabilities ("target" stream)
- drafter probabilities ("drafter" stream)
- their difference, target minus drafter ("delta" stream)

Notes
-----
- The feature dimension corresponds to vocabulary token ids, commonly the
  token ids (per-token-id statistics). When inputs have shape [N, D], D should
  be the feature size (e.g., vocab size), and each column aggregates
  statistics for a specific token id across observations (rows).
"""

from __future__ import annotations

from typing import Tuple
from enum import Enum

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

        logger.debug(
            "OnlineMeanStd.update: batch_count=%d, feature_dim=%d, prev_count=%d",
            batch_count,
            x64.size(1),
            self.count,
        )

        total_count = self.count + batch_count
        if total_count == 0:
            return

        delta = batch_mean - self.mean
        # Update mean and M2 using the parallel update formula
        self.mean = self.mean + delta * (batch_count / total_count)
        self.M2 = self.M2 + batch_M2 + delta * delta * (self.count * batch_count / total_count)
        self.count = total_count
        logger.debug("OnlineMeanStd.update: total_count=%d", self.count)

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

# Per-stream global accumulators.
_GLOBAL_STREAM_STATS: dict[str, OnlineMeanStd] = {}

# Sticky validity flags per stream. Once a stream is observed with valid
# probability distributions, subsequent invalid observations will raise.
_PROBS_VALID_SEEN_TRUE: dict[str, bool] = {}


class Stream(str, Enum):
    TARGET = "target"
    DRAFTER = "drafter"
    DELTA = "delta"


ALL_STREAMS: tuple[Stream, ...] = tuple(Stream)


def _normalize_stream(stream: "Stream | str") -> Stream:
    if isinstance(stream, Stream):
        return stream
    return Stream(stream)


def _get_stream(stream: "Stream | str") -> OnlineMeanStd:
    s = _normalize_stream(stream).value
    stats = _GLOBAL_STREAM_STATS.get(s)
    if stats is None:
        stats = OnlineMeanStd()
        _GLOBAL_STREAM_STATS[s] = stats
    return stats


@torch.no_grad()
def is_valid_probs(
    probs: torch.Tensor,
    stream: "Stream | str",
) -> bool:
    """Check whether probs is a valid probability matrix for a given stream.

    Valid means:
    - 2D tensor [N, D]
    - all finite
    - values in [0, 1]
    - row sums ≈ 1

    Sticky success: once this returns True for a stream, it must keep returning
    True for that stream. If a later call returns invalid, an exception is
    raised.
    """
    s = _normalize_stream(stream).value
    if probs.ndim != 2:
        raise ValueError("probs must be 2D [N, D]")

    is_finite = bool(torch.isfinite(probs).all().item())
    in_range = bool(((probs >= 0).all() and (probs <= 1).all()).item())
    row_sums = probs.sum(dim=-1)
    sums_close = bool(
        torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-6)
    )

    valid = is_finite and in_range and sums_close

    seen_true = _PROBS_VALID_SEEN_TRUE.get(s, False)
    if not valid:
        # Provide diagnostics, but avoid spamming at INFO level.
        try:
            min_sum = float(row_sums.min())
            max_sum = float(row_sums.max())
            mean_sum = float(row_sums.mean())
        except Exception:  # noqa: BLE001
            min_sum = max_sum = mean_sum = float("nan")
        logger.debug(
            "Invalid probs for stream '%s': is_finite=%s, in_range=%s, sums_close=%s, "
            "row_sum[min=%.6e, max=%.6e, mean=%.6e]",
            s,
            is_finite,
            in_range,
            sums_close,
            min_sum,
            max_sum,
            mean_sum,
        )
    if seen_true and not valid:
        raise RuntimeError(
            f"Probability matrix for stream '{s}' became invalid after previously being valid."
        )
    if valid and not seen_true:
        logger.debug("First valid probability matrix observed for stream '%s'", s)
        _PROBS_VALID_SEEN_TRUE[s] = True

    return valid


@torch.no_grad()
def update_global_probs_stats(
    probs_target: torch.Tensor,
    probs_drafter: torch.Tensor | None,
) -> None:
    """Update per-stream global statistics for target, drafter, and delta.

    Args:
        probs_target: Tensor of shape [N, D]. Rows are probability vectors for
            the target model. Expected to be in [0, 1] with row sums ≈ 1.
        probs_drafter: Tensor of shape [N, D]. Rows are probability vectors for
            the drafter model. Expected to be in [0, 1] with row sums ≈ 1.
    """
    assert probs_target.ndim == 2, "probs_target must be 2D [N, D]"
    if probs_drafter is not None:
        assert probs_drafter.ndim == 2, "probs_drafter must be 2D [N, D]"
        assert probs_target.shape == probs_drafter.shape, "shapes must match"

    streams_to_update: list[tuple[Stream, torch.Tensor]] = []

    # Validate and enqueue target
    logger.debug(f"{probs_target.shape=}")
    if is_valid_probs(probs_target, Stream.TARGET):
        streams_to_update.append((Stream.TARGET, probs_target))
    else:
        logger.debug("Skipping 'target' stream stats update: invalid probabilities (likely warmup)")

    # Validate drafter (if provided) and delta only if both are valid
    if probs_drafter is not None:
        logger.debug(f"{probs_drafter.shape=}")
        if is_valid_probs(probs_drafter, Stream.DRAFTER) and is_valid_probs(probs_target, Stream.TARGET):
            # Delta rows should sum to ~0. Use float64 and a slightly relaxed atol
            # to account for accumulation and prior tolerances on each stream.
            delta = probs_target - probs_drafter
            delta_row_sums = delta.to(torch.float64).sum(dim=-1)
            logger.debug(
                "delta_row_sums stats: min=%.6e, max=%.6e, mean=%.6e",
                float(delta_row_sums.min()),
                float(delta_row_sums.max()),
                float(delta_row_sums.mean()),
            )
            assert torch.allclose(
                delta_row_sums, torch.zeros_like(delta_row_sums), rtol=0, atol=3e-4
            )
            streams_to_update.extend([(Stream.DRAFTER, probs_drafter), (Stream.DELTA, delta)])
        else:
            logger.debug("Skipping 'drafter' and 'delta' stats update: invalid probabilities (likely warmup)")
    else:
        logger.debug("probs_drafter=None; updating only the 'target' stream")

    # Update per-stream accumulators
    for stream, x in streams_to_update:
        stats = _get_stream(stream)
        stats.update(x)
        try:
            mean, std = stats.get()
            logger.debug(
                "%s stats (global): count=%d, dim=%d, mean_mean=%.6f, std_mean=%.6f",
                _normalize_stream(stream).value,
                stats.count,
                mean.numel(),
                float(mean.mean()),
                float(std.mean()),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Global OnlineMeanStd get skipped for {_normalize_stream(stream).value}: {e}")

    # Persist per-process, per-stream accumulators
    stats_dir = os.environ.get("VLLM_PROBS_STATS_DIR", "probs_stats")
    try:
        os.makedirs(stats_dir, exist_ok=True)
        streams_to_persist = [s for s, _ in streams_to_update]
        for stream in streams_to_persist:
            stats = _get_stream(stream)
            stream_name = _normalize_stream(stream).value
            file_path = os.path.join(stats_dir, f"probs_stats_{stream_name}_{os.getpid()}.pt")
            payload = {
                "count": stats.count,
                "mean": stats.mean.detach().to(dtype=torch.float64, device="cpu")
                if stats.mean is not None else None,
                "M2": stats.M2.detach().to(dtype=torch.float64, device="cpu")
                if stats.M2 is not None else None,
            }
            torch.save(payload, file_path)
            logger.debug(
                "Saved probs stats to %s (stream=%s, count=%d, dim=%s)",
                file_path,
                stream_name,
                payload["count"],
                int(payload["mean"].numel()) if payload["mean"] is not None else None,
            )
    except Exception as save_e:  # noqa: BLE001
        logger.debug(f"Failed to persist global probs stats: {save_e}")


@torch.no_grad()
def get_global_probs_stats(stream: "Stream | str") -> tuple[torch.Tensor, torch.Tensor]:
    """Return the global (mean, std) per-token-id statistics for a stream.

    Args:
        stream: One of {"target", "drafter", "delta"}.
    """
    return _get_stream(stream).get()


@torch.no_grad()
def reset_global_probs_stats(stream: "Stream | str | None" = None) -> None:
    """Reset the global accumulators.

    Args:
        stream: If provided, reset only the specified stream. If None, reset
            all streams.
    """
    if stream is None:
        for s in list(_GLOBAL_STREAM_STATS.keys()):
            _GLOBAL_STREAM_STATS[s].reset()
    else:
        _get_stream(stream).reset()


@torch.no_grad()
def aggregate_saved_probs_stats(stats_dir: str, stream: "Stream | str") -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Aggregate per-process saved stats into global mean/std for a stream.

    The directory should contain files saved by this module with names like
    "probs_stats_{stream}_{pid}.pt" that include keys: count, mean, M2.

    Args:
        stats_dir: Directory containing saved per-process stats files.
        stream: One of {"target", "drafter", "delta"}.

    Returns:
        (mean, std, count): Aggregated tensors (float64 CPU) and total count.
    """
    s = _normalize_stream(stream).value
    # Flat layout (legacy and current)
    pattern = os.path.join(stats_dir, f"probs_stats_{s}_*.pt")
    files = sorted(glob.glob(pattern))
    logger.debug("Found %d saved stats files for stream '%s' in %s", len(files), s, stats_dir)
    if not files:
        raise ValueError(f"No stats files found for stream '{s}' in {stats_dir}")

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
                logger.debug("Skipping %s: missing or non-positive count/mean/M2", fp)
                continue
            mean = mean.to(dtype=torch.float64, device="cpu")
            M2 = M2.to(dtype=torch.float64, device="cpu")
            logger.debug("Loaded stats from %s: count=%d, dim=%d", fp, cnt, int(mean.numel()))
        except Exception:
            logger.debug("Failed to load stats from %s; skipping", fp)
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
    logger.info(
        "Aggregated stats for '%s': count=%d, dim=%d, mean=%.6f, std=%.6f",
        s,
        total_count,
        int(agg_mean.numel()),
        float(agg_mean.mean()),
        float(std.mean()),
    )
    return agg_mean, std, total_count


def visualize_per_token_stats(mean: torch.Tensor,
                              std: torch.Tensor | None,
                              output_path: str,
                              top_k: int = 50,
                              stream_name: str | None = None,
                              order_indices: torch.Tensor | None = None,
                              sort_info: str | None = None) -> str:
    """Visualize per-token-id mean (and optional std) statistics.

    This function is intended to be called once at the end of a benchmark.
    If matplotlib is available, it saves a figure; otherwise it falls back
    to saving a CSV file containing the top-K tokens by mean (and std if
    provided).

    Args:
        mean: 1D tensor of shape [feature_size], mean per token id.
        std: Optional 1D tensor of shape [feature_size], std per token id.
        output_path: Path to save the visualization (e.g., "probs_stats.png").
        top_k: Number of top token ids to show in the chart.

    Returns:
        The path of the created file (PNG or CSV).
    """
    if mean.dim() != 1:
        raise ValueError("visualize_per_token_stats expects mean of shape [token_id_size]")
    if std is not None and std.dim() != 1:
        raise ValueError("visualize_per_token_stats expects std of shape [token_id_size]")

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

        # Determine the ordering of token ids used for visualization.
        if order_indices is None:
            order_t = torch.argsort(mean_cpu, descending=True)
        else:
            # Use provided global order (already CPU-compatible or convertible)
            order_t = order_indices.detach().to(dtype=torch.long, device="cpu")
        # Select top_k entries according to the order
        sel_t = order_t[:top_k]
        mean_sorted = mean_cpu[sel_t].numpy()

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        x = np.arange(mean_sorted.shape[0])
        title_prefix = f"[{stream_name}] " if stream_name else ""
        sort_suffix = f" (order: {sort_info})" if sort_info else ""
        title = f"{title_prefix}Sorted per-token-id means{sort_suffix}"
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
            title = f"{title_prefix}Sorted per-token-id means with ±1 std deviation bars{sort_suffix}"

        ax.scatter(x, mean_sorted, color="navy", s=12, label="Mean", zorder=3)
        ax.set_title(title)
        if std_cpu is not None:
            ax.legend()

        ax.set_xlabel("Feature (sorted by mean)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        logger.debug(
            "Saving per-token stats plot to %s (stream=%s, top_k=%d, order=%s)",
            output_path,
            stream_name,
            top_k,
            sort_info,
        )
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
        if order_indices is None:
            order_t = torch.argsort(mean_cpu, descending=True)
        else:
            order_t = order_indices.detach().to(dtype=torch.long, device="cpu")
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
        logger.debug(
            "Matplotlib unavailable; wrote per-token stats CSV to %s (stream=%s, top_k=%d, order=%s)",
            csv_path,
            stream_name,
            top_k,
            sort_info,
        )
        return csv_path


@torch.no_grad()
def visualize_mean_std_correlation(mean: torch.Tensor,
                                   std: torch.Tensor,
                                   output_path: str,
                                   stream_name: str | None = None,
                                   order_indices: torch.Tensor | None = None,
                                   top_k: int | None = None) -> str:
    """Visualize correlation between per-token-id mean and std.

    Produces a scatter plot with x=mean, y=std for the selected token ids.
    If ``top_k`` is provided, selects the top-K tokens by the provided
    ``order_indices`` (or by descending mean if not provided). Falls back
    to writing a CSV with columns ``mean,std`` when plotting is unavailable.
    """
    if mean.dim() != 1 or std.dim() != 1:
        raise ValueError("visualize_mean_std_correlation expects 1D mean and std tensors")

    # Move to CPU float64 for stable plotting/saving.
    mean_cpu = mean.detach().to(dtype=torch.float64, device="cpu")
    std_cpu = std.detach().to(dtype=torch.float64, device="cpu")

    # Select subset if requested
    if top_k is not None and int(top_k) > 0 and int(top_k) < int(mean_cpu.numel()):
        if order_indices is None:
            order_t = torch.argsort(mean_cpu, descending=True)
        else:
            order_t = order_indices.detach().to(dtype=torch.long, device="cpu")
        sel_t = order_t[: int(top_k)]
        mean_sel = mean_cpu[sel_t]
        std_sel = std_cpu[sel_t]
    else:
        mean_sel = mean_cpu
        std_sel = std_cpu

    try:
        import os
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        x_vals = mean_sel.numpy()
        y_vals = std_sel.numpy()

        # Pearson correlation (guard small sample sizes)
        if x_vals.size >= 2:
            # np.corrcoef returns 2x2 matrix; [0,1] is the correlation
            with np.errstate(all="ignore"):
                pearson = float(np.corrcoef(x_vals, y_vals)[0, 1])
        else:
            pearson = float("nan")

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_vals, y_vals, s=8, color="purple", alpha=0.6)

        title_prefix = f"[{stream_name}] " if stream_name else ""
        subset_suffix = ""
        if top_k is not None and int(top_k) > 0 and int(top_k) < int(mean_cpu.numel()):
            subset_suffix = f" (top_k={int(top_k)})"
        ax.set_title(f"{title_prefix}Per-token mean vs std scatter{subset_suffix}\nPearson r={pearson:.4f}")
        ax.set_xlabel("Mean")
        ax.set_ylabel("Std")
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        logger.debug(
            "Saving mean-std correlation plot to %s (stream=%s, top_k=%s, pearson=%.6f)",
            output_path,
            stream_name,
            int(top_k) if top_k is not None else "all",
            pearson,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path
    except Exception:
        # Fallback: save CSV with mean and std pairs
        import os
        import numpy as np
        csv_path = output_path
        if csv_path.lower().endswith((".png", ".jpg", ".jpeg")):
            csv_path = os.path.splitext(csv_path)[0] + ".csv"
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        data = np.stack([mean_sel.numpy(), std_sel.numpy()], axis=1)
        header = "mean,std"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt=["%.10f", "%.10f"])
        logger.debug(
            "Matplotlib unavailable; wrote mean-std correlation CSV to %s (stream=%s, top_k=%s)",
            csv_path,
            stream_name,
            int(top_k) if top_k is not None else "all",
        )
        return csv_path


@torch.no_grad()
def visualize_streams_pairplot(stats_dir: str,
                               aggregated_stats: dict,
                               output_path: str,
                               top_k: int | None = None) -> str:
    """Create a pairplot across available per-token-id stats (mean/std) per stream.

    Constructs a per-token table of columns like: target_mean, target_std, drafter_mean,
    drafter_std, delta_mean, delta_std. Selects up to top_k tokens based on a
    global order (target mean if available, else first available column).
    """
    available_columns: dict[str, torch.Tensor] = {}
    for stream_name, stats in aggregated_stats.items():
        prefix = _normalize_stream(stream_name).value
        if "mean" in stats and "std" in stats:
            available_columns[f"{prefix}_mean"] = stats["mean"].detach().to(dtype=torch.float64, device="cpu")
            available_columns[f"{prefix}_std"] = stats["std"].detach().to(dtype=torch.float64, device="cpu")

    if not available_columns:
        raise ValueError("No aggregated stats found for any stream.")

    # Determine global order (defaults to target_mean, else first available)
    base_col = available_columns.get("target_mean", next(iter(available_columns.values())))
    order_indices = torch.argsort(base_col, descending=True)
    logger.debug("Pairplot: using %s for sorting order", "target_mean" if "target_mean" in available_columns else "first-available")

    vocab_size = int(next(iter(available_columns.values())).numel())
    k_val = vocab_size if (top_k is None) or (isinstance(top_k, int) and (top_k <= 0 or top_k >= vocab_size)) else int(top_k)
    sel_idx = order_indices[:k_val]

    # Stable column order
    ordered_keys = [
        "target_mean", "target_std", "drafter_mean", "drafter_std", "delta_mean", "delta_std"
    ]
    columns = [k for k in ordered_keys if k in available_columns]

    import numpy as np
    matrix = np.vstack([available_columns[c][sel_idx].numpy() for c in columns]).T

    try:
        import os
        import matplotlib
        matplotlib.use("Agg", force=True)
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(matrix, columns=columns)
        logger.info(
            "Rendering seaborn.pairplot (top_k=%d, cols=%d, n=%d)",
            k_val,
            len(columns),
            df.shape[0],
        )
        g = sns.pairplot(df, diag_kind="hist", plot_kws=dict(s=8, alpha=0.5))
        g.fig.suptitle(f"Per-token pairplot (top_k={k_val}, cols={len(columns)}, n={len(df)})", y=1.02)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        logger.debug("Saving streams pairplot to %s (top_k=%d, cols=%d, n=%d)", output_path, k_val, len(columns), len(df))
        g.savefig(output_path, dpi=200)
        plt.close(g.fig)
        return output_path
    except Exception as e:  # noqa: BLE001
        # Fallback: save CSV of the selected table
        import os
        csv_path = output_path
        if csv_path.lower().endswith((".png", ".jpg", ".jpeg")):
            csv_path = os.path.splitext(csv_path)[0] + ".csv"
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        header = ",".join(columns)
        np.savetxt(csv_path, matrix, delimiter=",", header=header, comments="")
        logger.debug("Seaborn/matplotlib unavailable; wrote pairplot CSV to %s (top_k=%d, cols=%d, n=%d)", csv_path, k_val, len(columns), matrix.shape[0])
        return csv_path


@torch.no_grad()
def visualize_streams_correlation_heatmap(stats_dir: str,
                                          aggregated_stats: dict,
                                          output_path: str,
                                          top_k: int | None = None,
                                          order_indices: torch.Tensor | None = None) -> str:
    """Create a 6x6 Pearson correlation heatmap across available stream stats.
    Columns considered: target_mean, target_std, drafter_mean, drafter_std,
    delta_mean, delta_std. Correlations are computed across token ids using
    all available rows (full vocab) on CPU float64.
    """
    available_columns: dict[str, torch.Tensor] = {}
    for stream_name, stats in aggregated_stats.items():
        prefix = _normalize_stream(stream_name).value
        if "mean" in stats and "std" in stats:
            available_columns[f"{prefix}_mean"] = stats["mean"].detach().to(dtype=torch.float64, device="cpu")
            available_columns[f"{prefix}_std"] = stats["std"].detach().to(dtype=torch.float64, device="cpu")

    if not available_columns:
        raise ValueError("No aggregated stats found for any stream.")

    # Determine global order if not provided
    if order_indices is None:
        base_col = available_columns.get("target_mean", next(iter(available_columns.values())))
        order_indices = torch.argsort(base_col, descending=True)
        logger.debug("Heatmap: using %s for sorting order", "target_mean" if "target_mean" in available_columns else "first-available")

    vocab_size = int(next(iter(available_columns.values())).numel())
    k_val = vocab_size if (top_k is None) or (isinstance(top_k, int) and (top_k <= 0 or top_k >= vocab_size)) else int(top_k)
    sel_idx = order_indices[:k_val]

    ordered_keys = [
        "target_mean", "target_std", "drafter_mean", "drafter_std", "delta_mean", "delta_std"
    ]
    columns = [k for k in ordered_keys if k in available_columns]

    import numpy as np
    import os
    # Build matrix [num_tokens, num_cols] for the selected top-k
    mat = np.vstack([available_columns[c][sel_idx].numpy() for c in columns]).T
    # Compute Pearson correlation matrix across columns
    with np.errstate(all="ignore"):
        corr = np.corrcoef(mat, rowvar=False)

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        sns.heatmap(corr,
                    xticklabels=columns,
                    yticklabels=columns,
                    vmin=-1.0,
                    vmax=1.0,
                    cmap="coolwarm",
                    annot=True,
                    fmt=".2f",
                    square=True,
                    ax=ax)
        
        title_suffix = f"(top_k={k_val})" if k_val < vocab_size else "(full vocab)"
        ax.set_title(f"Per-token Pearson correlation across streams {title_suffix}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        logger.debug("Saving streams correlation heatmap to %s", output_path)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path
    except Exception:
        # Fallback: write CSV of the correlation matrix
        csv_path = output_path
        if csv_path.lower().endswith((".png", ".jpg", ".jpeg")):
            csv_path = os.path.splitext(csv_path)[0] + ".csv"
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        header = ",".join(columns)
        np.savetxt(csv_path, corr, delimiter=",", header=header, comments="")
        logger.debug("Seaborn/matplotlib unavailable; wrote correlation heatmap CSV to %s", csv_path)
        return csv_path

def _generate_visualizations_for_stream(
    stats_dir: str,
    stream: "Stream | str",
    aggregated_stats: dict,
    mean: torch.Tensor,
    std: torch.Tensor,
    order_indices: torch.Tensor | None = None,
    sort_info: str | None = None,
    top_ks: list[int | None] | None = None,
    sort_scores_path: str | None = None,
    top_k_threshold: int | None = None,
) -> dict[str, str]:
    """Generate per-token-id visualizations for a single stream using pre-aggregated stats.
    """
    s = _normalize_stream(stream).value

    if top_ks is None:
        top_ks = [-1]

    mapping: dict[str, str] = {}
    num_created = 0
    vocab_size = int(mean.numel())
    for k in top_ks:
        k_val = vocab_size if (k is None) or (isinstance(k, int) and (k <= 0 or k >= vocab_size)) else int(k)
        suffix = 'all' if k_val == vocab_size else k_val
        out_path = os.path.join(stats_dir, f"probs_stats_{s}_top_k_{suffix}.png")
        created_path = visualize_per_token_stats(mean, std, out_path, top_k=k_val, stream_name=s, order_indices=order_indices, sort_info=sort_info)
        mapping[f"probs_stats/{s}/image_top_k_{suffix}"] = created_path
        num_created += 1
        logger.debug("Created visualization: key=%s, path=%s", f"probs_stats/{s}/image_top_k_{suffix}", created_path)

        if std is not None:
            corr_out_path = os.path.join(stats_dir, f"probs_stats_{s}_corr_top_k_{suffix}.png")
            corr_created_path = visualize_mean_std_correlation(
                mean,
                std,
                corr_out_path,
                stream_name=s,
                order_indices=order_indices,
                top_k=k_val,
            )
            mapping[f"probs_stats/{s}/corr_top_k_{suffix}"] = corr_created_path
            num_created += 1
            logger.debug("Created visualization: key=%s, path=%s", f"probs_stats/{s}/corr_top_k_{suffix}", corr_created_path)

        if s == Stream.TARGET.value:
            if top_k_threshold is not None and k_val > int(top_k_threshold):
                logger.info("Skipping pairplot for top_k=%d > threshold=%d", k_val, int(top_k_threshold))
                continue

            logger.info(
                "Starting pairplot for stream '%s' with top_k=%s (suffix=%s) -> generating %s",
                s,
                k_val,
                suffix,
                f"probs_stats_pairplot_top_k_{suffix}.png",
            )
            pairplot_path = os.path.join(stats_dir, f"probs_stats_pairplot_top_k_{suffix}.png")
            pairplot_created_path = visualize_streams_pairplot(
                stats_dir=stats_dir,
                aggregated_stats=aggregated_stats,
                output_path=pairplot_path,
                top_k=k_val,
            )
            mapping[f"probs_stats/pairplot_top_k_{suffix}"] = pairplot_created_path
            num_created += 1
            logger.debug("Created visualization: key=%s, path=%s", f"probs_stats/pairplot_top_k_{suffix}", pairplot_created_path)

            try:
                heatmap_path = os.path.join(stats_dir, f"probs_stats_corr_heatmap_top_k_{suffix}.png")
                heatmap_created_path = visualize_streams_correlation_heatmap(
                    stats_dir,
                    aggregated_stats,
                    heatmap_path,
                    top_k=k_val,
                    order_indices=order_indices,
                )
                mapping[f"probs_stats/corr_heatmap_top_k_{suffix}"] = heatmap_created_path
                num_created += 1
                logger.debug("Created visualization: key=%s, path=%s", f"probs_stats/corr_heatmap_top_k_{suffix}", heatmap_created_path)
            except Exception as e:
                logger.debug("Failed to create correlation heatmap for top_k=%s: %s", suffix, e)


    logger.info("Generated %d visualization artifacts for '%s' into %s", num_created, s, stats_dir)
    return mapping


def save_visualizations(
    stats_dir: str,
    top_ks: list[int | None] | None = None,
    sort_scores_path: str | None = None,
    top_k_threshold: int | None = None,
) -> dict[str, str]:
    """
    Generate and save all visualizations, aggregating stats once.
    """
    aggregated_stats = {}
    for stream in ALL_STREAMS:
        try:
            mean, std, count = aggregate_saved_probs_stats(stats_dir, stream)
            aggregated_stats[stream.value] = {"mean": mean, "std": std, "count": count}
        except Exception as e:
            logger.debug("Could not aggregate stats for stream '%s': %s", stream.value, e)

    if not aggregated_stats:
        logger.warning("No stream data to visualize in %s", stats_dir)
        return {}

    order_indices: torch.Tensor | None = None
    sort_info: str | None = None
    if sort_scores_path:
        try:
            scores = torch.load(sort_scores_path, map_location="cpu")
            if isinstance(scores, dict) and "tensor" in scores:
                scores = scores["tensor"]
            scores = torch.as_tensor(scores, dtype=torch.float64, device="cpu")
            vocab_size = int(next(iter(aggregated_stats.values()))['mean'].numel())
            if scores.dim() != 1 or int(scores.numel()) != vocab_size:
                raise ValueError("Sorting scores must be a 1D tensor with length == vocab size")
            order_indices = torch.argsort(scores, descending=True)
            sort_info = f"external:{os.path.basename(sort_scores_path)}"
            logger.info("Using external sorting scores at %s", sort_scores_path)
        except Exception as e:
            logger.warning("Failed to load sort scores from %s (%s). Falling back to target means.", sort_scores_path, e)
            order_indices = None
    
    if order_indices is None and "target" in aggregated_stats:
        target_mean = aggregated_stats["target"]["mean"]
        order_indices = torch.argsort(target_mean.detach().to(dtype=torch.float64, device="cpu"), descending=True)
        sort_info = "target-mean"
        logger.info("Using target stream mean for sorting order across streams.")
    elif order_indices is None:
        first_stream_mean = next(iter(aggregated_stats.values()))['mean']
        order_indices = torch.argsort(first_stream_mean.detach().to(dtype=torch.float64, device="cpu"), descending=True)
        sort_info = "first-available-stream-mean"
        logger.info("Target stream not found. Using first available stream for sorting order.")

    all_mappings = {}
    for stream_name, stats in aggregated_stats.items():
        stream_mapping = _generate_visualizations_for_stream(
            stats_dir=stats_dir,
            stream=stream_name,
            aggregated_stats=aggregated_stats,
            mean=stats["mean"],
            std=stats["std"],
            order_indices=order_indices,
            sort_info=sort_info,
            top_ks=top_ks,
            sort_scores_path=sort_scores_path,
            top_k_threshold=top_k_threshold,
        )
        all_mappings.update(stream_mapping)

    return all_mappings
