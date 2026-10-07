import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from thop import profile

from src.models.search_space import SearchSpaceCNN
from src.nas.search_space import (
    CHANNEL_OPTIONS,
    KERNEL_OPTIONS,
    POOLING_OPTIONS,
)


SEED = 42
NUM_CANDIDATES = 8
BATCH_SIZE = 128
EPOCHS = 2

TRAIN_SAMPLES = 10000
TEST_SAMPLES = 5000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_unique_architectures(num_candidates):
    all_architectures = []

    for channels in CHANNEL_OPTIONS:
        for kernel_size in KERNEL_OPTIONS:
            for use_pooling in POOLING_OPTIONS:
                all_architectures.append(
                    {
                        "channels": channels,
                        "kernel_size": kernel_size,
                        "use_pooling": use_pooling,
                    }
                )

    random.shuffle(all_architectures)

    return all_architectures[:num_candidates]


def measure_model(model, device):

    model.eval()

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    dummy_input = torch.randn(
        1, 3, 32, 32
    ).to(device)

    flops, _ = profile(
        model,
        inputs=(dummy_input,),
        verbose=False,
    )

    with torch.no_grad():

        for _ in range(10):
            model(dummy_input)

        start = time.perf_counter()

        for _ in range(100):
            model(dummy_input)

        end = time.perf_counter()

    latency_ms = (
        (end - start) / 100
    ) * 1000

    model_size_mb = (
        parameters * 4 / (1024 ** 2)
    )

    return {
        "parameters": parameters,
        "flops": int(flops),
        "model_size_mb": round(model_size_mb, 4),
        "latency_ms": round(latency_ms, 4),
    }


def train_model(
    model,
    train_loader,
    test_loader,
    device,
):

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

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1,
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_accuracy = (
            100 * correct / total
        )

        print(
            f"    Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {running_loss / len(train_loader):.4f} "
            f"- Train accuracy: {train_accuracy:.2f}%"
        )

    # Test accuracy

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1,
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    test_accuracy = (
        100 * correct / total
    )

    return test_accuracy


def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

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

    architectures = generate_unique_architectures(
        NUM_CANDIDATES
    )

    print(
        f"\nEvaluating {NUM_CANDIDATES} unique architectures..."
    )

    results = []

    for index, architecture in enumerate(
        architectures,
        start=1,
    ):

        print("\n" + "=" * 60)

        print(
            f"Candidate {index}/{NUM_CANDIDATES}"
        )

        print(
            f"Architecture: {architecture}"
        )

        model = SearchSpaceCNN(
            channels=architecture["channels"],
            kernel_size=architecture["kernel_size"],
            use_pooling=architecture["use_pooling"],
        ).to(device)

        metrics = measure_model(
            model,
            device,
        )

        print(
            f"Parameters: {metrics['parameters']:,}"
        )

        print(
            f"FLOPs: {metrics['flops']:,}"
        )

        print(
            f"Latency: {metrics['latency_ms']:.4f} ms"
        )

        print("Training...")

        accuracy = train_model(
            model,
            train_loader,
            test_loader,
            device,
        )

        print(
            f"Test accuracy: {accuracy:.2f}%"
        )

        results.append(
            {
                "candidate": index,
                "channels": str(
                    architecture["channels"]
                ),
                "kernel_size": architecture[
                    "kernel_size"
                ],
                "use_pooling": architecture[
                    "use_pooling"
                ],
                "accuracy_percent": round(
                    accuracy,
                    2,
                ),
                "parameters": metrics[
                    "parameters"
                ],
                "flops": metrics["flops"],
                "model_size_mb": metrics[
                    "model_size_mb"
                ],
                "latency_ms": metrics[
                    "latency_ms"
                ],
                "device": str(device),
            }
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/nas_accuracy_results.csv",
        index=False,
    )

    print("\n" + "=" * 60)

    print(
        "\nFinal NAS results:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        "results/nas_accuracy_results.csv"
    )


if __name__ == "__main__":
    main()