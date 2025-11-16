"""Utilities to benchmark Aurora's LoRA configurations.

The script generates random batches that follow the :class:`aurora.batch.Batch` API,
so it can be run without access to ERA5 files.  It reports

- the total number of parameters and trainable parameters,
- the average step time over a few synthetic optimisation steps, and
- the peak GPU memory (if CUDA is used),

for different combinations of `use_lora` and `lora_mode`.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import torch

from aurora import AuroraPretrained, Batch, Metadata


SURFACE_VARS = ("2t", "10u", "10v", "msl")
STATIC_VARS = ("lsm", "z", "slt")
ATMOS_VARS = ("z", "u", "v", "t", "q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of synthetic optimisation steps per configuration.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Synthetic batch size.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=2,
        help="Number of history steps in the synthetic batch.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=33,
        help="Latitudinal resolution of the synthetic batch.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Longitudinal resolution of the synthetic batch.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Number of atmospheric pressure levels.",
    )
    parser.add_argument(
        "--lora-steps",
        type=int,
        default=40,
        help="Maximum number of rollout steps a LoRA adapter is applied for.",
    )
    parser.add_argument(
        "--lora-modes",
        nargs="+",
        default=("single", "from_second", "all"),
        choices=("single", "from_second", "all"),
        help="LoRA rollout modes to benchmark when LoRA is enabled.",
    )
    parser.add_argument(
        "--include-no-lora",
        action="store_true",
        help="Also benchmark a configuration with LoRA disabled.",
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="Load the default pretrained checkpoint before benchmarking.",
    )
    parser.add_argument(
        "--use-autocast",
        action="store_true",
        help="Enable autocast inside Aurora for lower memory usage.",
    )
    return parser.parse_args()


def ensure_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def random_metadata(
    height: int,
    width: int,
    batch_size: int,
    num_levels: int,
    device: torch.device,
) -> Metadata:
    """Create metadata tensors consistent with the synthetic batch."""
    if num_levels <= 0:
        raise ValueError("`num_levels` must be positive.")
    lat = torch.linspace(90.0, -90.0, height, device=device)
    lon = torch.linspace(0.0, 360.0, width + 1, device=device)[:-1]
    start = datetime(2020, 1, 1, 0, 0)
    times = tuple(start + timedelta(hours=6 * i) for i in range(batch_size))
    levels = tuple(
        float(x) for x in torch.linspace(50.0, 1000.0, num_levels, device=device).cpu().tolist()
    )
    return Metadata(
        lat=lat,
        lon=lon,
        time=times,
        atmos_levels=levels,
    )


def synthetic_batch(
    *,
    batch_size: int,
    history: int,
    height: int,
    width: int,
    num_levels: int,
    device: torch.device,
) -> Batch:
    """Generate a random batch compatible with Aurora."""
    metadata = random_metadata(height, width, batch_size, num_levels, device)
    surf_vars = {
        name: torch.randn(batch_size, history, height, width, device=device)
        for name in SURFACE_VARS
    }
    static_vars = {name: torch.randn(height, width, device=device) for name in STATIC_VARS}
    atmos_vars = {
        name: torch.randn(batch_size, history, num_levels, height, width, device=device)
        for name in ATMOS_VARS
    }
    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata,
    )


def latitude_weights(latitudes: torch.Tensor) -> torch.Tensor:
    """Compute cosine latitude weights normalised to an average weight of 1."""
    weights = torch.cos(latitudes * torch.pi / 180.0).clamp(min=0.0)
    return weights / weights.mean()


def latitude_weighted_loss(pred: Batch) -> torch.Tensor:
    """Latitude-weighted MSE on the SST proxy (2m temperature)."""
    sst = pred.surf_vars["2t"]
    weights = latitude_weights(pred.metadata.lat.to(sst.device, dtype=sst.dtype)).view(
        1, 1, -1, 1
    )
    mse = (weights * sst).pow(2).mean()
    return mse


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def log_model_stats(name: str, total: int, trainable: int) -> None:
    print(
        f"[{name}] total_params={total/1e6:.2f}M, "
        f"trainable_params={trainable/1e6:.2f}M"
    )


def benchmark_configuration(
    *,
    use_lora: bool,
    lora_mode: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float | int | str | None]:
    """Run a short synthetic training loop and collect metrics."""
    model = AuroraPretrained(
        use_lora=use_lora,
        lora_mode=lora_mode,
        lora_steps=args.lora_steps,
        autocast=args.use_autocast,
    )
    if args.load_checkpoint:
        model.load_checkpoint(strict=False)
    model.configure_activation_checkpointing()
    model.train()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    total_params, trainable_params = count_parameters(model)
    log_model_stats(
        name=f"use_lora={use_lora}, mode={lora_mode}",
        total=total_params,
        trainable=trainable_params,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step_times: list[float] = []
    for _ in range(args.steps):
        batch = synthetic_batch(
            batch_size=args.batch_size,
            history=args.history,
            height=args.height,
            width=args.width,
            num_levels=args.levels,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pred = model(batch)
        loss = latitude_weighted_loss(pred)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - start)

    avg_time = sum(step_times) / len(step_times)
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )
    return {
        "use_lora": use_lora,
        "lora_mode": lora_mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "avg_step_time_s": avg_time,
        "peak_mem_mb": peak_mem_mb,
    }


def print_results(results: list[dict[str, float | int | str | None]]) -> None:
    headers = [
        "use_lora",
        "lora_mode",
        "total_params(M)",
        "trainable_params(M)",
        "avg_step_time(s)",
        "peak_mem(MiB)",
    ]
    print("\n" + "-" * 80)
    print("{:<10} {:<12} {:>18} {:>20} {:>18} {:>15}".format(*headers))
    for row in results:
        peak_mem = (
            f"{row['peak_mem_mb']:.1f}" if isinstance(row["peak_mem_mb"], (int, float)) else "n/a"
        )
        print(
            f"{str(row['use_lora']):<10} "
            f"{str(row['lora_mode']):<12} "
            f"{row['total_params']/1e6:>18.2f} "
            f"{row['trainable_params']/1e6:>20.2f} "
            f"{row['avg_step_time_s']:>18.3f} "
            f"{peak_mem:>15}"
        )
    print("-" * 80)


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    configs: list[tuple[bool, str]] = [(True, mode) for mode in args.lora_modes]
    if args.include_no_lora:
        configs.append((False, "n/a"))

    results = []
    for use_lora, mode in configs:
        print(f"\nRunning configuration use_lora={use_lora}, lora_mode={mode}")
        metrics = benchmark_configuration(
            use_lora=use_lora,
            lora_mode=mode if use_lora else "single",
            device=device,
            args=args,
        )
        if not use_lora:
            metrics["lora_mode"] = "n/a"
        results.append(metrics)

    print_results(results)


if __name__ == "__main__":
    main()
