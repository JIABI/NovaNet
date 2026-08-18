#!/usr/bin/env python3
"""Measure the current NovaNet forward pass with an auditable protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from novanet.config import NovaNetConfig, load_config
from novanet.model import NovaNet


WORKER_MARKER = "NOVANET_BENCHMARK_RESULT="


def parse_candidate_caps(value: str) -> list[int]:
    try:
        candidates = [
            int(item.strip()) for item in value.split(",") if item.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "candidate caps must be comma-separated integers"
        ) from error
    if not candidates or any(item < 2 for item in candidates):
        raise argparse.ArgumentTypeError(
            "candidate caps must contain integers greater than or equal to two"
        )
    if len(set(candidates)) != len(candidates):
        raise argparse.ArgumentTypeError("candidate caps must be unique")
    return candidates


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _percentile(ordered: list[float], probability: float) -> float:
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    index = int(math.ceil(probability * len(ordered))) - 1
    return float(ordered[min(max(index, 0), len(ordered) - 1)])


def _process_peak_rss() -> tuple[int | None, str]:
    """Return the worker-process high-water RSS with explicit semantics."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError):
        return None, "unavailable_on_this_platform"
    if sys.platform == "darwin":
        return value, "getrusage.ru_maxrss_bytes"
    return value * 1024, "getrusage.ru_maxrss_kib_converted_to_bytes"


def _total_system_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _cpu_model_name() -> str:
    if sys.platform == "darwin":
        try:
            apple_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if apple_model:
                return apple_model
        except (OSError, subprocess.SubprocessError):
            pass
    reported = platform.processor().strip()
    if reported:
        return reported
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return "unreported"


def _hardware_metadata(requested_device: str) -> dict[str, Any]:
    device = _resolve_device(requested_device)
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model_name(),
        "logical_cpu_count": os.cpu_count(),
        "total_system_memory_bytes": _total_system_memory_bytes(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "requested_device": requested_device,
        "resolved_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata["accelerator"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [properties.major, properties.minor],
        }
    elif device.type == "mps":
        metadata["accelerator"] = {
            "name": "Apple Metal Performance Shaders",
            "total_memory_bytes": None,
        }
    else:
        metadata["accelerator"] = None
    return metadata


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_case(
    base: NovaNetConfig,
    *,
    candidates: int,
    warmup: int,
    repetitions: int,
    threads: int,
    requested_device: str,
    seed: int,
) -> dict[str, Any]:
    """Measure one candidate cap inside one otherwise fresh worker process."""

    if candidates < 2:
        raise ValueError("candidates must be at least two")
    if warmup < 0 or repetitions < 1 or threads < 1:
        raise ValueError("warmup must be nonnegative; repetitions/threads positive")
    torch.set_num_threads(threads)
    torch.manual_seed(seed + candidates)
    device = _resolve_device(requested_device)
    config = replace(
        base,
        experiment=replace(
            base.experiment,
            candidate_cap=candidates,
            num_satellites=max(candidates, base.experiment.num_satellites),
        ),
    )
    model = NovaNet(config).to(device).eval()
    horizon = config.planner.horizon_steps
    features = config.model.node_feature_dim

    cpu_generator = torch.Generator().manual_seed(seed + candidates)
    node = torch.randn(
        1, horizon, candidates, features, generator=cpu_generator
    )
    # Future recurrent inputs contain only the first five fields.  Keeping
    # field six at zero follows the deployed input contract without altering
    # the tensor shape or computational workload.
    node[:, 1:, :, 5] = 0.0
    prior = torch.ones(candidates, candidates) - torch.eye(candidates)
    prior = prior / prior.sum(dim=-1, keepdim=True)
    adjacency = prior.reshape(1, 1, candidates, candidates).expand(
        1, horizon, candidates, candidates
    ).clone()
    valid = torch.ones(1, horizon, candidates, dtype=torch.bool)
    incumbent = torch.zeros(1, dtype=torch.long)
    freeze = torch.zeros(1, dtype=torch.long)
    ttl = (
        torch.rand(
            1, horizon, candidates, generator=cpu_generator
        )
        * config.planner.ttl_reference_s
    )
    nominal_sinr = torch.randn(
        1, horizon, candidates, generator=cpu_generator
    ) + 8.0
    inputs = (node, adjacency, valid, incumbent, freeze, ttl, nominal_sinr)
    inputs = tuple(tensor.to(device) for tensor in inputs)
    node, adjacency, valid, incumbent, freeze, ttl, nominal_sinr = inputs

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    parameter_bytes = sum(_tensor_bytes(parameter) for parameter in model.parameters())
    buffer_bytes = sum(_tensor_bytes(buffer) for buffer in model.buffers())
    input_bytes = sum(_tensor_bytes(tensor) for tensor in inputs)

    with torch.inference_mode():
        for _ in range(warmup):
            model(
                node,
                adjacency,
                valid,
                incumbent,
                ttl,
                nominal_sinr,
                initial_freeze=freeze,
            )
        _synchronize(device)
        accelerator_memory: dict[str, Any] = {}
        if device.type == "cuda":
            before_allocated = int(torch.cuda.memory_allocated(device))
            before_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
            accelerator_memory.update(
                {
                    "allocated_before_timing_bytes": before_allocated,
                    "reserved_before_timing_bytes": before_reserved,
                    "measurement": "PyTorch CUDA allocator",
                }
            )
        elif device.type == "mps" and hasattr(torch, "mps"):
            accelerator_memory.update(
                {
                    "allocated_before_timing_bytes": int(
                        torch.mps.current_allocated_memory()
                    ),
                    "driver_before_timing_bytes": int(
                        torch.mps.driver_allocated_memory()
                    ),
                    "measurement": (
                        "PyTorch MPS current allocation; peak allocator metric "
                        "is unavailable"
                    ),
                }
            )

        samples_ms: list[float] = []
        for _ in range(repetitions):
            _synchronize(device)
            start = time.perf_counter_ns()
            model(
                node,
                adjacency,
                valid,
                incumbent,
                ttl,
                nominal_sinr,
                initial_freeze=freeze,
            )
            _synchronize(device)
            samples_ms.append((time.perf_counter_ns() - start) / 1e6)

        if device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            accelerator_memory.update(
                {
                    "peak_allocated_during_timing_bytes": peak_allocated,
                    "peak_reserved_during_timing_bytes": peak_reserved,
                    "incremental_peak_allocated_bytes": max(
                        0, peak_allocated - before_allocated
                    ),
                }
            )
        elif device.type == "mps" and hasattr(torch, "mps"):
            accelerator_memory.update(
                {
                    "allocated_after_timing_bytes": int(
                        torch.mps.current_allocated_memory()
                    ),
                    "driver_after_timing_bytes": int(
                        torch.mps.driver_allocated_memory()
                    ),
                }
            )

    peak_rss_bytes, peak_rss_method = _process_peak_rss()
    ordered = sorted(samples_ms)
    mean_ms = statistics.fmean(samples_ms)
    return {
        "candidate_cap": candidates,
        "case_config_fingerprint": config.fingerprint,
        "batch_size": 1,
        "horizon_steps": horizon,
        "node_feature_dim": features,
        "future_recurrent_feature_dim": 5,
        "dtype": str(node.dtype).replace("torch.", ""),
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "timing": {
            "mean_ms": mean_ms,
            "median_ms": statistics.median(samples_ms),
            "p95_ms": _percentile(ordered, 0.95),
            "p99_ms": _percentile(ordered, 0.99),
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
            "population_std_ms": statistics.pstdev(samples_ms),
            "inferences_per_second_from_mean": 1000.0 / mean_ms,
            "clock": "time.perf_counter_ns",
            "quantile_method": "nearest-rank",
            "outlier_filter": "none; every timed forward is retained",
            "device_synchronized_per_sample": device.type in {"cuda", "mps"},
        },
        "memory": {
            "model_parameter_bytes": parameter_bytes,
            "model_buffer_bytes": buffer_bytes,
            "logical_input_tensor_bytes": input_bytes,
            "worker_process_peak_rss_bytes": peak_rss_bytes,
            "worker_process_peak_rss_method": peak_rss_method,
            "worker_process_peak_rss_scope": (
                "isolated worker high-water RSS including Python, PyTorch, "
                "model, inputs, warm-up, and timed forwards; not incremental "
                "model-only memory"
            ),
            "accelerator_allocator": accelerator_memory or None,
        },
    }


def _worker_result(args: argparse.Namespace) -> int:
    base = load_config(args.config)
    row = benchmark_case(
        base,
        candidates=args._worker_candidate,
        warmup=args.warmup,
        repetitions=args.repetitions,
        threads=args.threads,
        requested_device=args.device,
        seed=args.seed,
    )
    print(WORKER_MARKER + json.dumps(row, separators=(",", ":")))
    return 0


def _run_isolated_case(args: argparse.Namespace, candidates: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        args.config,
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
        "--threads",
        str(args.threads),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--_worker-candidate",
        str(candidates),
    ]
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark worker failed for K={candidates}:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(WORKER_MARKER):
            return json.loads(line[len(WORKER_MARKER) :])
    raise RuntimeError(
        f"benchmark worker for K={candidates} returned no result marker"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the current 6D NovaNet model and Soft-DP."
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument(
        "--candidate-caps",
        type=parse_candidate_caps,
        default=parse_candidate_caps("8,16,32"),
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "PyTorch intra-op CPU threads; the unchanged inter-op count is "
            "recorded separately."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, cuda, mps, or auto; cpu is the reproducible default.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--output", default="results/benchmark/inference_current.json"
    )
    parser.add_argument(
        "--_worker-candidate", type=int, default=None, help=argparse.SUPPRESS
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 1 or args.threads < 1:
        parser.error("warmup must be nonnegative; repetitions/threads positive")
    if args._worker_candidate is not None:
        return _worker_result(args)

    base = load_config(args.config)
    if base.model.node_feature_dim != 6:
        raise ValueError("The current manuscript model must use six node features")
    rows = [
        _run_isolated_case(args, candidates)
        for candidates in args.candidate_caps
    ]
    parameter_counts = {row["parameters"] for row in rows}
    if len(parameter_counts) != 1:
        raise RuntimeError("Model parameter count unexpectedly varies with K_cand")

    script_path = Path(__file__).resolve()
    config_path = Path(args.config)
    payload = {
        "artifact_status": "current_measured",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "batch-1 NovaNet neural encoder, learned adjacency, residual/HOF "
            "heads, energy construction, and freeze-aware Soft-DP"
        ),
        "excluded_from_timing": [
            "TLE/SGP4 propagation",
            "candidate construction",
            "feature construction",
            "checkpoint I/O",
            "multi-UE coordination",
        ],
        "weights": (
            "randomly initialized current architecture; weights do not change "
            "operation shapes or parameter/memory counts"
        ),
        "case_isolation": (
            "Each candidate cap runs in a fresh worker process so process peak "
            "RSS is not inherited from a preceding cap."
        ),
        "requested_intraop_threads": args.threads,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "canonical_config_path": str(config_path),
        "canonical_config_sha256": _file_sha256(config_path),
        "canonical_config_fingerprint": base.fingerprint,
        "benchmark_script_sha256": _file_sha256(script_path),
        "model_contract": {
            "current_node_feature_dim": base.model.node_feature_dim,
            "future_recurrent_feature_dim": 5,
            "horizon_steps": base.planner.horizon_steps,
            "parameters": next(iter(parameter_counts)),
        },
        "hardware": _hardware_metadata(args.device),
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
