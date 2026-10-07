import time
import torch
import pandas as pd
from thop import profile

from src.models.search_space import SearchSpaceCNN
from src.nas.search_space import sample_architecture


def evaluate_architecture(architecture):
    """
    Build and evaluate one CNN architecture.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SearchSpaceCNN(
        channels=architecture["channels"],
        kernel_size=architecture["kernel_size"],
        use_pooling=architecture["use_pooling"],
    ).to(device)

    model.eval()

    # Count parameters
    parameters = sum(p.numel() for p in model.parameters())

    # Calculate FLOPs
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    flops, _ = profile(
        model,
        inputs=(dummy_input,),
        verbose=False,
    )

    # Measure inference latency
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

        start = time.perf_counter()

        for _ in range(100):
            model(dummy_input)

        end = time.perf_counter()

    latency_ms = ((end - start) / 100) * 1000

    # Approximate model size
    model_size_mb = parameters * 4 / (1024 ** 2)

    return {
        "channels": str(architecture["channels"]),
        "kernel_size": architecture["kernel_size"],
        "use_pooling": architecture["use_pooling"],
        "parameters": parameters,
        "flops": int(flops),
        "model_size_mb": round(model_size_mb, 4),
        "latency_ms": round(latency_ms, 4),
        "device": str(device),
    }


def main():

    print("Evaluating candidate architectures...\n")

    results = []

    for i in range(10):

        architecture = sample_architecture()

        metrics = evaluate_architecture(architecture)

        metrics["candidate"] = i + 1

        results.append(metrics)

        print(f"Candidate {i + 1}")
        print(f"Architecture: {architecture}")
        print(f"Parameters: {metrics['parameters']:,}")
        print(f"FLOPs: {metrics['flops']:,}")
        print(f"Model size: {metrics['model_size_mb']} MB")
        print(f"Latency: {metrics['latency_ms']} ms")
        print()

    df = pd.DataFrame(results)

    df.to_csv(
        "results/nas_candidates.csv",
        index=False,
    )

    print("Results saved to:")
    print("results/nas_candidates.csv")


if __name__ == "__main__":
    main()