import torch, numpy as np
torch.backends.cudnn.benchmark = False  # keep determinism off/on per need

def bench(M, N, K, iters=5000, dtype=torch.float32, tf32=True):
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    a = torch.randn(M, K, device='cuda', dtype=dtype)
    b = torch.randn(K, N, device='cuda', dtype=dtype)

    # Warm-up
    for _ in range(200):
        (a @ b).sum().backward() if a.requires_grad else a @ b
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        c = a @ b
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))  # ms

    arr = np.array(times)
    return {
        'mean_ms': arr.mean(),
        'p50_ms': np.percentile(arr, 50),
        'p95_ms': np.percentile(arr, 95),
        'p99_ms': np.percentile(arr, 99),
        'max_ms': arr.max(),
    }