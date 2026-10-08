
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results/nas_accuracy_results.csv")
PARETO_PATH = Path("results/pareto_front.csv")
PLOT_PATH = Path("results/figures/pareto_accuracy_vs_flops.png")


def find_pareto_front(df):
    """Maximize accuracy while minimizing FLOPs."""
    pareto_mask = []

    for _, candidate in df.iterrows():
        dominated = (
            (df["accuracy_percent"] >= candidate["accuracy_percent"])
            & (df["flops"] <= candidate["flops"])
            & (
                (df["accuracy_percent"] > candidate["accuracy_percent"])
                | (df["flops"] < candidate["flops"])
            )
        ).any()

        pareto_mask.append(not dominated)

    return df.loc[pareto_mask].sort_values("flops")


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RESULTS_PATH}. Run nas_accuracy.py first."
        )

    df = pd.read_csv(RESULTS_PATH)
    pareto = find_pareto_front(df)

    PARETO_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pareto.to_csv(PARETO_PATH, index=False)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        df["flops"] / 1_000_000,
        df["accuracy_percent"],
        label="All candidates",
    )
    plt.scatter(
        pareto["flops"] / 1_000_000,
        pareto["accuracy_percent"],
        marker="*",
        s=220,
        label="Pareto-optimal",
    )

    for _, row in df.iterrows():
        plt.annotate(
            str(int(row["candidate"])),
            (row["flops"] / 1_000_000, row["accuracy_percent"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("FLOPs (millions)")
    plt.ylabel("Test accuracy (%)")
    plt.title("CarbonNAS: Accuracy vs Computational Cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print("\nPareto-optimal candidates:")
    pareto[
        [
            "candidate",
            "accuracy_percent",
            "flops",
            "parameters",
            "latency_ms",
        ]
    ].to_string(index=False)

    print(f"\nSaved Pareto results to: {PARETO_PATH}")
    print(f"Saved plot to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
