
import time
import random
from pathlib import Path
import statistics

import torch
import pandas as pd
from thop import profile

from src.models.search_space import SearchSpaceCNN
from src.nas.search_space import sample_architecture


# Benchmark settings
WARMUP_RUNS = 20
REPEATS = 7
RUNS_PER_REPEAT = 100
INPUT_SHAPE = (1, 3, 32, 32)

# Keep CPU benchmarking conditions consistent.
if not torch.cuda.is_available():
    torch.set_num_threads(1)


def evaluate_architecture(architecture):
    """Measure model size, FLOPs, and repeated inference latency."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SearchSpaceCNN(
        channels=architecture["channels"],
        kernel_size=architecture["kernel_size"],
        use_pooling=architecture["use_pooling"],
    ).to(device)

    model.eval()
    dummy_input = torch.randn(*INPUT_SHAPE, device=device)

    # Count trainable and non-trainable parameters.
    parameters = sum(p.numel() for p in model.parameters())

    # Estimate FLOPs using THOP.
    with torch.no_grad():
        flops, _ = profile(
            model,
            inputs=(dummy_input,),
            verbose=False,
        )

    # Warm up the model before timing.
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(dummy_input)

        # CUDA operations are asynchronous, so synchronize if using a GPU.
        if device.type == "cuda":
            torch.cuda.synchronize()

        repeat_times_ms = []

        for _ in range(REPEATS):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            for _ in range(RUNS_PER_REPEAT):
                model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            latency_per_run_ms = (
                elapsed / RUNS_PER_REPEAT
            ) * 1000

            repeat_times_ms.append(latency_per_run_ms)

    median_latency_ms = statistics.median(repeat_times_ms)

    # Population standard deviation describes variability across repeats.
    latency_std_ms = statistics.pstdev(repeat_times_ms)

    # Approximate FP32 parameter storage, not total runtime memory.
    model_size_mb = parameters * 4 / (1024 ** 2)

    return {
        "channels": str(architecture["channels"]),
        "kernel_size": architecture["kernel_size"],
        "use_pooling": architecture["use_pooling"],
        "parameters": parameters,
        "flops": int(flops),
        "model_size_mb": round(model_size_mb, 4),
        "latency_ms": round(median_latency_ms, 4),
        "latency_std_ms": round(latency_std_ms, 4),
        "latency_repeats": REPEATS,
        "runs_per_repeat": RUNS_PER_REPEAT,
        "device": str(device),
    }


def main():
    random.seed(42)
    torch.manual_seed(42)

    print("Evaluating candidate architectures...")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Repeated measurements: {REPEATS}")
    print(f"Inference runs per repeat: {RUNS_PER_REPEAT}\n")

    results = []
    seen_architectures = set()

    # Try to collect 10 unique architectures.
    max_attempts = 100
    attempts = 0
    target_candidates = 10

    while (
        len(results) < target_candidates
        and attempts < max_attempts
    ):
        attempts += 1
        architecture = sample_architecture()

        architecture_key = (
            tuple(architecture["channels"]),
            architecture["kernel_size"],
            architecture["use_pooling"],
        )

        if architecture_key in seen_architectures:
            continue

        seen_architectures.add(architecture_key)

        metrics = evaluate_architecture(architecture)
        metrics["candidate"] = len(results) + 1
        results.append(metrics)

        print(f"Candidate {metrics['candidate']}")
        print(f"Architecture: {architecture}")
        print(f"Parameters: {metrics['parameters']:,}")
        print(f"FLOPs: {metrics['flops']:,}")
        print(f"Model size: {metrics['model_size_mb']} MB")
        print(f"Median latency: {metrics['latency_ms']} ms")
        print(f"Latency standard deviation: "
              f"{metrics['latency_std_ms']} ms\n")

    if not results:
        raise RuntimeError("No architectures were evaluated.")

    output_path = Path("results/nas_candidates.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"Evaluated {len(results)} unique architectures.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
