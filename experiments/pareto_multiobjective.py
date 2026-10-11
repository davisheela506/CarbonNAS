
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = Path("results/nas_multiobjective_results.csv")
OUTPUT_CSV = Path("results/nas_pareto_front_3d.csv")
OUTPUT_PLOT = Path(
    "results/figures/pareto_accuracy_flops_latency.png"
)


def find_pareto_front(df):
    """
    Maximize accuracy while minimizing FLOPs and latency.
    A candidate is dominated if another candidate is at least
    as good on every objective and strictly better on one.
    """

    accuracy = df["accuracy_percent"].to_numpy()
    flops = df["flops"].to_numpy()
    latency = df["latency_ms"].to_numpy()

    is_pareto = []

    for i in range(len(df)):
        dominated = (
            (accuracy >= accuracy[i])
            & (flops <= flops[i])
            & (latency <= latency[i])
            & (
                (accuracy > accuracy[i])
                | (flops < flops[i])
                | (latency < latency[i])
            )
        )

        is_pareto.append(not dominated.any())

    return df.loc[is_pareto].copy()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input results not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "candidate",
        "accuracy_percent",
        "flops",
        "latency_ms",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input results file is empty.")

    if df[list(required)].isnull().any().any():
        raise ValueError(
            "Missing values found in required Pareto objectives."
        )

    pareto = find_pareto_front(df)

    # Sort by computational cost to make the output easier to read.
    pareto = pareto.sort_values("flops")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    pareto.to_csv(OUTPUT_CSV, index=False)

    print("\nThree-objective Pareto front")
    print(
        pareto[
            [
                "candidate",
                "accuracy_percent",
                "flops",
                "latency_ms",
                "latency_std_ms",
            ]
        ].to_string(index=False)
    )

    print(f"\nCandidates evaluated: {len(df)}")
    print(f"Pareto-optimal candidates: {len(pareto)}")
    print(f"CSV saved to: {OUTPUT_CSV}")

    # Plot accuracy against FLOPs; use color to show latency.
    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        df["flops"] / 1_000_000,
        df["accuracy_percent"],
        c=df["latency_ms"],
        s=75,
        alpha=0.65,
        label="All candidates",
    )

    ax.scatter(
        pareto["flops"] / 1_000_000,
        pareto["accuracy_percent"],
        s=150,
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        label="Pareto front",
    )

    for _, row in df.iterrows():
        ax.annotate(
            str(int(row["candidate"])),
            (
                row["flops"] / 1_000_000,
                row["accuracy_percent"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Median inference latency (ms)")

    ax.set_xlabel("FLOPs (millions)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(
        "CarbonNAS: Accuracy, Computation, and Latency Trade-offs"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
