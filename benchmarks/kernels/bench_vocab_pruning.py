"""
Vector–matrix matmul latency benchmark for vocab pruning scenarios.

Shapes:
- vector: [batch, hidden]
- matrix: [hidden, vocab]

CLI example:
python benchmarks/kernels/bench_vocab_pruning.py \
  --batches 1 16 64 \
  --hiddens 1024 4096 \
  --vocabs 32000 131072 \
  --dtype bf16 \
  --iters 3000 --warmup-iters 300
"""

import argparse
import itertools
import torch
import numpy as np

torch.backends.cudnn.benchmark = False


DTYPE_MAP = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}


def set_matmul_precision(allow_tf32: bool) -> None:
    torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')


def make_inputs(batch: int, hidden: int, vocab: int, dtype: torch.dtype, device: str):
    a = torch.randn(batch, hidden, device=device, dtype=dtype)
    b = torch.randn(hidden, vocab, device=device, dtype=dtype)
    return a, b


def build_run_fn(backend: str, a: torch.Tensor, b: torch.Tensor):
    if backend == 'torch':
        return lambda: a @ b
    raise ValueError(f"Unsupported backend: {backend}")


def warmup(run_once, iters: int = 200) -> None:
    for _ in range(iters):
        run_once()
    torch.cuda.synchronize()


def time_many(run_once, iters: int) -> np.ndarray:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        run_once()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return np.array(times, dtype=np.float64)


def summarize(times_ms: np.ndarray) -> dict:
    return {
        'mean_ms': float(times_ms.mean()),
        'p50_ms': float(np.percentile(times_ms, 50)),
        'p95_ms': float(np.percentile(times_ms, 95)),
        'p99_ms': float(np.percentile(times_ms, 99)),
        'min_ms': float(times_ms.min()),
        'max_ms': float(times_ms.max()),
    }


def bench_vm(
    batch: int,
    hidden: int,
    vocab: int,
    *,
    iters: int = 2000,
    warmup_iters: int = 200,
    dtype: torch.dtype = torch.float32,
    backend: str = 'torch',
    tf32: bool = True,
    device: str = 'cuda',
) -> dict:
    set_matmul_precision(tf32)
    a, b = make_inputs(batch, hidden, vocab, dtype, device)
    run_once = build_run_fn(backend, a, b)
    warmup(run_once, warmup_iters)
    times = time_many(run_once, iters)
    return summarize(times)


def run_sweep(
    batches: list[int],
    hiddens: list[int],
    vocabs: list[int],
    *,
    dtype_name: str = 'bf16',
    backend: str = 'torch',
    iters: int = 2000,
    warmup_iters: int = 200,
    tf32: bool = True,
) -> None:
    dtype = DTYPE_MAP[dtype_name]
    for batch, hidden, vocab in itertools.product(batches, hiddens, vocabs):
        stats = bench_vm(
            batch,
            hidden,
            vocab,
            iters=iters,
            warmup_iters=warmup_iters,
            dtype=dtype,
            backend=backend,
            tf32=tf32,
        )
        print(
            f"batch={batch} hidden={hidden} vocab={vocab} dtype={dtype_name} backend={backend} "
            f"p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms p99={stats['p99_ms']:.3f}ms "
            f"mean={stats['mean_ms']:.3f}ms min={stats['min_ms']:.3f}ms max={stats['max_ms']:.3f}ms"
        )


# Back-compat thin wrapper for programmatic use with (M, N, K)
def bench(M, N, K, iters=5000, dtype=torch.float32, tf32=True):
    stats = bench_vm(M, K, N, iters=iters, dtype=dtype, tf32=tf32)
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vector-Matrix matmul latency benchmark')
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 16, 64, 128, 256, 512, 1024, 4096, 65536])
    parser.add_argument('--hiddens', type=int, nargs='+', default=[1024, 4096])
    parser.add_argument('--vocabs', type=int, nargs='+', default=[1024, 4096, 16384, 32000, 32768, 128000, 131072])
    parser.add_argument('--dtype', type=str, choices=list(DTYPE_MAP.keys()), default='bf16')
    parser.add_argument('--backend', type=str, choices=['torch'], default='torch')
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--warmup-iters', type=int, default=300)
    parser.add_argument('--no-tf32', action='store_true', help='Disable TF32 matmul on Ampere+')
    args = parser.parse_args()

    run_sweep(
        args.batches,
        args.hiddens,
        args.vocabs,
        dtype_name=args.dtype,
        backend=args.backend,
        iters=args.iters,
        warmup_iters=args.warmup_iters,
        tf32=not args.no_tf32,
    )