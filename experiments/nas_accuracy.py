
import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.search_space import SearchSpaceCNN


SEED = 42
BATCH_SIZE = 128
EPOCHS = 2

TRAIN_SAMPLES = 10000
TEST_SAMPLES = 5000

BENCHMARK_PATH = Path("results/nas_candidates.csv")
OUTPUT_PATH = Path("results/nas_multiobjective_results.csv")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_benchmarked_architectures():
    """Load architectures and their existing benchmark measurements."""

    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_PATH}\n"
            "Run python -m src.nas.evaluate_candidates first."
        )

    df = pd.read_csv(BENCHMARK_PATH)

    required_columns = {
        "candidate",
        "channels",
        "kernel_size",
        "use_pooling",
        "parameters",
        "flops",
        "model_size_mb",
        "latency_ms",
        "latency_std_ms",
        "device",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Benchmark CSV is missing columns: {sorted(missing)}"
        )

    architectures = []

    for _, row in df.iterrows():
        channels = ast.literal_eval(row["channels"])

        architectures.append({
            "candidate": int(row["candidate"]),
            "channels": tuple(channels),
            "kernel_size": int(row["kernel_size"]),
            "use_pooling": (
                str(row["use_pooling"]).strip().lower() == "true"
                if isinstance(row["use_pooling"], str)
                else bool(row["use_pooling"])
            ),
            "parameters": int(row["parameters"]),
            "flops": int(row["flops"]),
            "model_size_mb": float(row["model_size_mb"]),
            "latency_ms": float(row["latency_ms"]),
            "latency_std_ms": float(row["latency_std_ms"]),
            "latency_repeats": int(row["latency_repeats"]),
            "runs_per_repeat": int(row["runs_per_repeat"]),
            "device": str(row["device"]),
        })

    if not architectures:
        raise ValueError("No architectures found in benchmark CSV.")

    candidate_ids = [a["candidate"] for a in architectures]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate IDs found in benchmark CSV.")

    return architectures


def prepare_datasets():
    """Use fixed CIFAR-10 subsets and identical preprocessing."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    print("\nLoading CIFAR-10...")

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_subset = Subset(
        train_dataset,
        range(TRAIN_SAMPLES),
    )

    test_subset = Subset(
        test_dataset,
        range(TEST_SAMPLES),
    )

    return train_subset, test_subset


def train_model(model, train_loader, test_loader, device):
    """Train one architecture and evaluate test accuracy."""

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

        print(
            f"    Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {running_loss / len(train_loader):.4f} "
            f"- Train accuracy: {train_accuracy:.2f}%"
        )

    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    architectures = load_benchmarked_architectures()

    print(f"Loaded {len(architectures)} benchmarked architectures.")
    print(f"Training settings: {EPOCHS} epochs, batch size {BATCH_SIZE}")
    print(
        f"Dataset subsets: {TRAIN_SAMPLES} training images, "
        f"{TEST_SAMPLES} test images"
    )

    train_subset, test_subset = prepare_datasets()

    results = []

    for architecture in architectures:
        candidate_id = architecture["candidate"]

        print("\n" + "=" * 65)
        print(f"Candidate {candidate_id}/{len(architectures)}")
        print(
            f"Architecture: channels={architecture['channels']}, "
            f"kernel={architecture['kernel_size']}, "
            f"pooling={architecture['use_pooling']}"
        )

        # Reset the seed for reproducible initialization and data order
        # for each candidate.
        set_seed(SEED + candidate_id)

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        model = SearchSpaceCNN(
            channels=architecture["channels"],
            kernel_size=architecture["kernel_size"],
            use_pooling=architecture["use_pooling"],
        ).to(device)

        print(f"Parameters: {architecture['parameters']:,}")
        print(f"FLOPs: {architecture['flops']:,}")
        print(f"Median benchmark latency: {architecture['latency_ms']:.4f} ms")
        print("Training...")

        accuracy = train_model(
            model,
            train_loader,
            test_loader,
            device,
        )

        print(f"Test accuracy: {accuracy:.2f}%")

        results.append({
            **architecture,
            "accuracy_percent": round(accuracy, 2),
            "training_epochs": EPOCHS,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "batch_size": BATCH_SIZE,
            "training_device": str(device),
        })

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 65)
    print("Unified multi-objective results:")
    print(
        results_df[
            [
                "candidate",
                "accuracy_percent",
                "parameters",
                "flops",
                "model_size_mb",
                "latency_ms",
                "latency_std_ms",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
