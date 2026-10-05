import os
import time
import csv

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from thop import profile

from src.models.cnn import TinyCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "results/tiny_cnn.pt"
RESULTS_PATH = "results/architectures.csv"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def evaluate_accuracy(model, test_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def measure_latency(model, input_tensor, iterations=100):
    model.eval()

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            model(input_tensor)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(iterations):
            model(input_tensor)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

    total_time = end - start
    return (total_time / iterations) * 1000


def main():
    print(f"Device: {DEVICE}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    model = TinyCNN(num_classes=10).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)

    accuracy = evaluate_accuracy(model, test_loader)

    parameters = count_parameters(model)

    model_size_mb = get_model_size_mb(model)

    dummy_input = torch.randn(
        1, 3, 32, 32,
        device=DEVICE
    )

    flops, params_from_thop = profile(
        model,
        inputs=(dummy_input,),
        verbose=False,
    )

    latency_ms = measure_latency(
        model,
        dummy_input
    )

    result = {
        "model": "TinyCNN",
        "accuracy_percent": round(accuracy, 2),
        "parameters": parameters,
        "model_size_mb": round(model_size_mb, 4),
        "flops": int(flops),
        "latency_ms": round(latency_ms, 4),
        "device": str(DEVICE),
    }

    os.makedirs("results", exist_ok=True)

    file_exists = os.path.exists(RESULTS_PATH)

    with open(
        RESULTS_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    print("\nEvaluation results")
    print("------------------")

    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()