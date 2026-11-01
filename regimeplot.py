
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_FOLDER = Path("./testcases")
OUTPUT_FILE = "all_qubits_diff_with_regimes.png"

# QUBIT_COUNTS = [5,6,7,8,9] use this for the plot with number of nodes. 
QUBIT_COUNTS = [25, 50, 75, 100, 125]

SHEETS = [
    "Wide Random",
    "High Fidelity",
    "Mid-Noise",
    "Random Bottleneck 30%"
]

REGIME_LABELS = [
    "Wide Random",
    "High Reliability",
    "Mid-Noise",
    "Random Bottleneck 30%"
]

ALGORITHMS = [
    "DP",
    "Our Algo",
    "Baseline (CBT, Equal Alloc)"
]

# ------------------------------------------------------------
# Find and sort Excel files
# ------------------------------------------------------------

excel_files = [
    f for f in INPUT_FOLDER.glob("*.xlsx")
    if not f.name.startswith("~$")
]

def get_qubit_count(file):
    digits = "".join(filter(str.isdigit, file.stem))

    if not digits:
        raise ValueError(
            f"Cannot determine qubit count from filename: {file.name}"
        )

    return int(digits)

files_by_qubits = {
    get_qubit_count(f): f for f in excel_files
}

missing = [
    q for q in QUBIT_COUNTS
    if q not in files_by_qubits
]

if missing:
    raise FileNotFoundError(
        f"Missing Excel files for qubit counts: {missing}. "
        f"Expected files such as 25qubits.xlsx, 50qubits.xlsx, etc."
    )

excel_files = [
    files_by_qubits[q] for q in QUBIT_COUNTS
]

# ------------------------------------------------------------
# Calculate average differences from DP
# ------------------------------------------------------------

results = {}

for input_file, qubits in zip(excel_files, QUBIT_COUNTS):

    results[qubits] = {}

    for sheet in SHEETS:

        try:
            df = pd.read_excel(
                input_file,
                sheet_name=sheet
            )

        except ValueError:
            print(
                f"Warning: Missing sheet '{sheet}' "
                f"in {input_file.name}"
            )

            results[qubits][sheet] = {
                "Our Algo": np.nan,
                "Baseline (CBT, EA)": np.nan
            }
            continue

        # Fill down test case identifiers.
        df["Test Case"] = df["Test Case"].ffill()

        df = df[df["Algorithm"].isin(ALGORITHMS)]

        # One row per test case, one column per algorithm.
        pivot = df.pivot_table(
            index="Test Case",
            columns="Algorithm",
            values="Expected Entanglement",
            aggfunc="first"
        )

        pivot = pivot.reindex(columns=ALGORITHMS)

        # Difference from DP.
        pivot["Our Algo Diff"] = (
            pivot["DP"] - pivot["Our Algo"]
        )

        pivot["Baseline Diff"] = (
            pivot["DP"]
            - pivot["Baseline (CBT, Equal Alloc)"]
        )

        our_diff = pivot["Our Algo Diff"].mean()
        baseline_diff = pivot["Baseline Diff"].mean()

        results[qubits][sheet] = {
            "Our Algo": our_diff,
            "Baseline (CBT, EA)": baseline_diff
        }

        print(
            f"{qubits} qubits | {sheet}: "
            f"Our Algo = {our_diff:.6f}, "
            f"Baseline = {baseline_diff:.6f}, "
            f"Runs = {len(pivot)}"
        )

# ------------------------------------------------------------
# Plot: Four subplots, one per regime
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 9),
    sharex=False
)

axes = axes.flatten()

x = np.arange(len(QUBIT_COUNTS))
width = 0.30

for i, (sheet, regime_label) in enumerate(
    zip(SHEETS, REGIME_LABELS)
):

    ax = axes[i]

    our_values = [
        results[q][sheet]["Our Algo"]
        for q in QUBIT_COUNTS
    ]

    baseline_values = [
        results[q][sheet]["Baseline (CBT, EA)"]
        for q in QUBIT_COUNTS
    ]

    bars_ours = ax.bar(
        x - width / 2,
        our_values,
        width,
        label="Our Algo",
        hatch="///"
    )

    bars_baseline = ax.bar(
        x + width / 2,
        baseline_values,
        width,
        label="Baseline (CBT, EA)",
        hatch=".."
    )

    ax.set_title(
        regime_label,
        fontsize=15,
        fontweight="bold",
        pad=10
    )

    # Show qubit counts on every subplot.
    ax.set_xticks(x)
    ax.set_xticklabels(
        QUBIT_COUNTS,
        fontsize=12,
        fontweight="bold"
    )

    ax.tick_params(axis="y", labelsize=11)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.set_xlabel(
        "Number of Qubits",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Average Difference from DP",
        fontsize=12,
        fontweight="bold"
    )

    ax.axhline(
        y=0,
        color="black",
        linewidth=0.8
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    # Add numerical labels to bars.
    all_values = our_values + baseline_values
    finite_values = [
        v for v in all_values if np.isfinite(v)
    ]

    if finite_values:
        ymin = min(0, min(finite_values))
        ymax = max(0, max(finite_values))
        span = ymax - ymin

        if span == 0:
            span = 0.01

        ax.set_ylim(
            ymin - 0.12 * span,
            ymax + 0.25 * span
        )

        for bars in [bars_ours, bars_baseline]:
            for bar in bars:
                height = bar.get_height()

                if not np.isfinite(height):
                    continue

                ax.annotate(
                    f"{height:.3f}",
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        height
                    ),
                    xytext=(
                        0,
                        3 if height >= 0 else -3
                    ),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=8,
                    fontweight="bold"
                )

# ------------------------------------------------------------
# Shared legend and figure title
# ------------------------------------------------------------

handles, labels = axes[0].get_legend_handles_labels()
print(handles, labels)
fig.legend(
    handles,
    ['BoC-Alg', 'Baseline (CBT, EA)'],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    fontsize=12,
    frameon=True
)

fig.suptitle(
    "Average Difference from DP across 100 Instances",
    fontsize=18,
    fontweight="bold",
    y=1.06
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nCombined plot saved as: {OUTPUT_FILE}")
