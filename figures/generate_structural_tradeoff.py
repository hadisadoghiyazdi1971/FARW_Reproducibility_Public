"""Generate the publication structural coverage--fragility figure.

Deterministic values come from the frozen 160-recording structural summary;
Random values come from the saved-index, 30-seed structural evaluation.
No selector or structural metric is recomputed here.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOURCE_CSV = (
    PROJECT_ROOT
    / "reference_results"
    / "structural_deterministic_source.csv"
)
RANDOM_30SEED_CSV = (
    PROJECT_ROOT / "reference_results"
    / "random_30seed_structural_means.csv"
)
OUTPUT_PDF = SCRIPT_DIR / "Figure_6_structural_tradeoff_final.pdf"
OUTPUT_PNG = SCRIPT_DIR / "Figure_6_structural_tradeoff_final.png"

BUDGETS = (0.1, 0.2, 0.3, 0.4, 0.5)
METHOD_ORDER = ("farw", "medoid", "centroid", "renyi", "random")

# Frozen visual identity for subsequent manuscript figures.  Colours are from
# the colour-blind-friendly Okabe--Ito palette, supplemented by neutral grey.
STYLES = {
    "farw": {
        "label": "FARW",
        "color": "#D55E00",
        "marker": "o",
        "linestyle": "-",
    },
    "medoid": {
        "label": "Global Medoid",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-.",
    },
    "centroid": {
        "label": "Global Centroid",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "--",
    },
    "renyi": {
        "label": "Rényi RDS/RDS-IS",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
    },
    "random": {
        "label": "Random",
        "color": "#595959",
        "marker": "X",
        "linestyle": (0, (5, 2)),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_means() -> dict[str, dict[float, tuple[float, float]]]:
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f"Frozen structural summary not found: {SOURCE_CSV}")

    data: dict[str, dict[float, tuple[float, float]]] = {
        method: {} for method in METHOD_ORDER
    }
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method_key"].strip().lower()
            if method == "random":
                raise ValueError("The deterministic structural source must not contain Random rows")
            if method not in data:
                continue
            budget = float(row["budget"])
            if budget not in BUDGETS:
                continue
            if int(row["recording_count"]) != 160:
                raise ValueError(
                    f"Expected 160 recordings for {method}, budget={budget:g}."
                )
            if budget in data[method]:
                raise ValueError(f"Duplicate row for {method}, budget={budget:g}.")
            data[method][budget] = (
                float(row["C_normalized_mean"]),
                float(row["fragility_F1_normalized_mean"]),
            )

    missing = [
        (method, budget)
        for method in METHOD_ORDER[:-1]
        for budget in BUDGETS
        if budget not in data[method]
    ]
    if missing:
        raise ValueError(f"Missing frozen method-budget rows: {missing}")

    if not RANDOM_30SEED_CSV.is_file():
        raise FileNotFoundError(f"30-seed Random structural summary missing: {RANDOM_30SEED_CSV}")
    expected_seeds = ";".join(str(seed) for seed in range(1, 31))
    with RANDOM_30SEED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        random_rows = list(csv.DictReader(handle))
    if len(random_rows) != len(BUDGETS):
        raise ValueError("Expected five 30-seed Random structural rows")
    seen_budgets: set[float] = set()
    for row in random_rows:
        budget = float(row["budget"])
        if (
            row["method_key"] != "random"
            or budget not in BUDGETS
            or budget in seen_budgets
            or int(row["seed_count"]) != 30
            or row["seed_set"] != expected_seeds
            or int(row["recording_count"]) != 160
            or int(row["case_count"]) != 4800
        ):
            raise ValueError(f"Invalid 30-seed Random structural row: {row}")
        seen_budgets.add(budget)
        data["random"][budget] = (
            float(row["C_normalized_mean"]),
            float(row["fragility_F1_normalized_mean"]),
        )
    if seen_budgets != set(BUDGETS):
        raise ValueError("Incomplete 30-seed Random structural grid")

    # Scientific gates requested for this figure.
    for budget in BUDGETS:
        coverage = {method: data[method][budget][0] for method in METHOD_ORDER}
        fragility = {method: data[method][budget][1] for method in METHOD_ORDER}
        min_coverage = min(coverage.values())
        min_fragility = min(fragility.values())
        coverage_winners = [m for m, value in coverage.items() if value == min_coverage]
        fragility_winners = [m for m, value in fragility.items() if value == min_fragility]
        if coverage_winners != ["centroid"]:
            raise ValueError(
                f"Coverage gate failed at {budget:.0%}: {coverage_winners}"
            )
        if fragility_winners != ["farw"]:
            raise ValueError(
                f"Fragility gate failed at {budget:.0%}: {fragility_winners}"
            )

    return data


def configure_typography() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.55,
            "lines.markersize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="y", which="major", color="#D9DEE5", linewidth=0.55, alpha=0.75)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#50565E")
    axis.spines["bottom"].set_color("#50565E")
    axis.tick_params(direction="out", length=3.0, width=0.7, colors="#30343B")
    axis.set_xlim(8, 52)
    axis.set_xticks((10, 20, 30, 40, 50))
    axis.set_xlabel("Retention (%)")


def plot_figure(data: dict[str, dict[float, tuple[float, float]]]) -> None:
    configure_typography()
    retention = [100.0 * budget for budget in BUDGETS]
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 3.58), sharex=True)

    for axis in axes:
        style_axis(axis)

    handles = []
    for method in METHOD_ORDER:
        style = STYLES[method]
        coverage = [data[method][budget][0] for budget in BUDGETS]
        fragility = [data[method][budget][1] for budget in BUDGETS]
        common = {
            "color": style["color"],
            "marker": style["marker"],
            "linestyle": style["linestyle"],
            "label": style["label"],
            "markeredgecolor": style["color"],
            "markeredgewidth": 0.8,
            "markerfacecolor": "white" if method != "farw" else style["color"],
            "zorder": 4 if method == "farw" else 3,
        }
        line = axes[0].plot(retention, coverage, **common)[0]
        axes[1].plot(retention, fragility, **common)
        handles.append(line)

    title_y = 1.095
    axes[0].set_title(
        "(a)  Nominal coverage", loc="left", y=title_y, fontweight="bold", pad=0
    )
    axes[1].set_title(
        "(b)  Singleton fragility", loc="left", y=title_y, fontweight="bold", pad=0
    )
    axes[0].set_ylabel(r"Normalised coverage cost, $C/s_r$")
    axes[1].set_ylabel(r"Normalised singleton fragility, $F_1/s_r$")

    # The limits include the full observed range with substantial surrounding
    # context; they resolve the curves without implying a zero-based bar scale.
    axes[0].set_ylim(0.17, 0.49)
    axes[0].set_yticks((0.20, 0.25, 0.30, 0.35, 0.40, 0.45))
    axes[1].set_ylim(0.0020, 0.0109)
    axes[1].set_yticks((0.002, 0.004, 0.006, 0.008, 0.010))
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value * 1e3:g}")
    )
    axes[1].text(
        0.0,
        1.015,
        r"$\times 10^{-3}$",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#30343B",
    )

    figure.legend(
        handles=handles,
        labels=[STYLES[method]["label"] for method in METHOD_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        handlelength=2.25,
        handletextpad=0.45,
        columnspacing=1.05,
    )
    figure.subplots_adjust(left=0.085, right=0.988, top=0.84, bottom=0.245, wspace=0.265)

    fixed_time = datetime(2026, 9, 15, 0, 0, tzinfo=timezone.utc)
    pdf_metadata = {
        "Title": "Global structural performance of window-selection methods",
        "Author": "FARW study",
        "Subject": "Normalised coverage cost and singleton fragility",
        "Keywords": "FARW, coverage, singleton fragility, window selection",
        "Creator": "generate_structural_tradeoff.py",
        "CreationDate": fixed_time,
        "ModDate": fixed_time,
    }
    figure.savefig(OUTPUT_PDF, format="pdf", metadata=pdf_metadata)
    figure.savefig(
        OUTPUT_PNG,
        format="png",
        dpi=600,
        metadata={"Software": "generate_structural_tradeoff.py"},
    )
    plt.close(figure)


def main() -> None:
    data = load_frozen_means()
    plot_figure(data)
    print(f"Source: {SOURCE_CSV}")
    print(f"Source SHA-256: {sha256(SOURCE_CSV)}")
    print(f"Random 30-seed source: {RANDOM_30SEED_CSV}")
    print(f"Random 30-seed SHA-256: {sha256(RANDOM_30SEED_CSV)}")
    print("Verification: Centroid minimum coverage 5/5; FARW minimum fragility 5/5")
    print(f"Vector PDF: {OUTPUT_PDF}")
    print(f"PNG preview: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
