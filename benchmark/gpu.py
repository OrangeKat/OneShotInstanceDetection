"""GPU configuration, memory accounting, and cleanup."""

from __future__ import annotations

import gc
from logging import Logger

import torch


def configure_gpu(fraction: float, log: Logger) -> None:
    """Cap this process's GPU memory so a runaway model cannot hang the driver.

    Args:
        fraction: Requested share of total VRAM, or ``0`` to auto-size from
            the currently free memory.
        log: Logger.
    """
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    if fraction > 0:
        budget = min(fraction * total, 0.9 * free)
    else:
        budget = min(0.9 * total, 0.9 * free)
    budget = min(max(budget, 512 * 1024**2), total)
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    log.info(
        "GPU memory budget: %.1f GiB (%.1f GiB free of %.1f GiB)",
        budget / 1024**3,
        free / 1024**3,
        total / 1024**3,
    )
    if free < 2 * 1024**3:
        log.warning(
            "less than 2 GiB of GPU memory is free; close other GPU apps or "
            "lower --max-side"
        )


def release_gpu() -> None:
    """Drop model references and return cached blocks to the driver."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_gpu_peak() -> None:
    """Reset the GPU peak-memory counters used in reports."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_memory_stats() -> dict[str, float]:
    """Return peak GPU memory for the current model, in MiB."""
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_gpu_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 1024**2,
            1,
        ),
    }
