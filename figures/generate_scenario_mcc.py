"""Regenerate Figure 7 with the final 30-seed Random baseline.

No classifier or selector is executed. Deterministic MCC values and FULL
references are read from the frozen Unified-3F results, while Random points are
read from the completed 30-seed aggregate. The seed-wise worst-MCC diagnostic
is also validated so that the figure and manuscript table use their intended,
different aggregation orders.
"""

from __future__ import annotations

import csv
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FROZEN_SOURCE = (
    PROJECT_ROOT
    / "reference_results"
    / "classification_unified_3f_source.csv"
)
RANDOM_SCENARIO_SOURCE = (
    PROJECT_ROOT
    / "reference_results"
    / "random_30seed_scenario_mean.csv"
)
RANDOM_WORST_SOURCE = (
    PROJECT_ROOT
    / "reference_results"
    / "random_30seed_seedwise_worst_diagnostic.csv"
)
OUTPUT_PDF = SCRIPT_DIR / "Figure_7_scenario_mcc.pdf"
OUTPUT_PNG = SCRIPT_DIR / "Figure_7_scenario_mcc.png"

EXPECTED_SHA256 = {
    FROZEN_SOURCE: "877245b3b2c14e2adc44f92f6b7eaf2e7baa0ab7c35aed6887d5becebf587cf1",
    RANDOM_SCENARIO_SOURCE: "96938012b0f52cbab0cd2781f0621b385a97d810f4ad43e2a28cff6448e1bb7e",
    RANDOM_WORST_SOURCE: "3f9dc8c7c9c35c1cbfd3a8d6e1f8c0effb7975bbeabc12b8cfc68fe4e8face7e",
}

SCENARIOS = (1, 2, 3)
BUDGETS = (0.1, 0.2, 0.3, 0.4, 0.5)
K_BY_BUDGET = {0.1: 35, 0.2: 70, 0.3: 105, 0.4: 140, 0.5: 175}
DETERMINISTIC_METHODS = ("farw", "medoid", "centroid", "renyi")
METHOD_ORDER = DETERMINISTIC_METHODS + ("random",)
EXPECTED_SEEDS = tuple(range(1, 31))
SCENARIO_TITLES = {
    1: "Intermediate Depth of Cut",
    2: "Unseen Speed–Feed",
    3: "Cross-Geometry",
}
Y_LIMITS = (0.84, 0.93)
Y_TICKS = (0.84, 0.86, 0.88, 0.90, 0.92)

EXPECTED_DETERMINISTIC_6DP = {
    1: {
        "farw": (0.920544, 0.910264, 0.915956, 0.911839, 0.909384),
        "medoid": (0.910036, 0.919129, 0.915585, 0.916779, 0.927541),
        "centroid": (0.904019, 0.909808, 0.912341, 0.911263, 0.911916),
        "renyi": (0.924516, 0.916962, 0.909629, 0.911156, 0.924475),
    },
    2: {
        "farw": (0.860951, 0.896238, 0.913647, 0.916891, 0.918249),
        "medoid": (0.854292, 0.858876, 0.925059, 0.927024, 0.923230),
        "centroid": (0.915639, 0.875865, 0.899209, 0.891158, 0.925482),
        "renyi": (0.854047, 0.859467, 0.922784, 0.922498, 0.892620),
    },
    3: {
        "farw": (0.864692, 0.890107, 0.887890, 0.866505, 0.883625),
        "medoid": (0.885779, 0.888991, 0.900564, 0.876419, 0.868285),
        "centroid": (0.873804, 0.870284, 0.879979, 0.879840, 0.869553),
        "renyi": (0.876618, 0.895463, 0.885499, 0.879575, 0.877866),
    },
}
EXPECTED_RANDOM = {
    1: (0.9035189537532423, 0.9126405904662274, 0.9161349963512516,
        0.9182754327852813, 0.9192314865526641),
    2: (0.8472109256803224, 0.8659060565779406, 0.8791229219371919,
        0.8825545408139750, 0.8857021230305266),
    3: (0.8798888488382236, 0.8849559906243664, 0.8839181827310633,
        0.8852711381222498, 0.8830383624188113),
}
EXPECTED_RANDOM_WORST = (
    0.8429687748451199,
    0.8606272374739018,
    0.8680971655316156,
    0.8691318124490905,
    0.8705302539363264,
)
EXPECTED_FULL_6DP = {1: 0.925228, 2: 0.924325, 3: 0.877541}
EXPECTED_LEADERS = {
    0.1: "centroid",
    0.2: "farw",
    0.3: "medoid",
    0.4: "centroid",
    0.5: "farw",
}

STYLES = {
    "farw": {"label": "FARW", "color": "#D55E00", "marker": "o", "linestyle": "-"},
    "medoid": {"label": "Global Medoid", "color": "#009E73", "marker": "^", "linestyle": "-."},
    "centroid": {"label": "Global Centroid", "color": "#0072B2", "marker": "s", "linestyle": "--"},
    "renyi": {"label": "Rényi RDS/RDS-IS", "color": "#CC79A7", "marker": "D", "linestyle": ":"},
    "random": {"label": "Random", "color": "#595959", "marker": "X", "linestyle": (0, (5, 2))},
}
FULL_STYLE = {"label": "FULL (unreduced)", "color": "#222222"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != EXPECTED_SHA256[path]:
        raise ValueError(f"SHA-256 mismatch for authoritative source: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_seed_set(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(";") if value)


def load_and_validate() -> tuple[
    dict[int, dict[str, dict[float, float]]], dict[int, float], dict[float, float]
]:
    frozen_rows = read_csv(FROZEN_SOURCE)
    if len(frozen_rows) != 63:
        raise ValueError(f"Expected 63 final deterministic/FULL rows; found {len(frozen_rows)}.")

    reduced = {
        scenario: {method: {} for method in METHOD_ORDER}
        for scenario in SCENARIOS
    }
    full: dict[int, float] = {}
    for row in frozen_rows:
        scenario = int(row["scenario"])
        method = row["method_key"].strip().lower()
        value = float(row["MCC"])
        if method == "random":
            raise ValueError("The public deterministic source must not contain Random rows")
        if method == "full":
            if scenario in full:
                raise ValueError(f"Duplicate FULL row for Scenario {scenario}.")
            full[scenario] = value
            continue
        if scenario not in SCENARIOS or method not in DETERMINISTIC_METHODS:
            raise ValueError(f"Unexpected frozen row: scenario={scenario}, method={method}")
        budget = float(row["budget"])
        if budget not in BUDGETS or budget in reduced[scenario][method]:
            raise ValueError(f"Invalid or duplicate frozen row: {row}")
        reduced[scenario][method][budget] = value

    if set(full) != set(SCENARIOS):
        raise ValueError("Expected exactly one FULL reference per scenario.")
    for scenario in SCENARIOS:
        for method in DETERMINISTIC_METHODS:
            if set(reduced[scenario][method]) != set(BUDGETS):
                raise ValueError(f"Incomplete deterministic grid: S{scenario}, {method}")
            observed = tuple(
                round(reduced[scenario][method][budget], 6) for budget in BUDGETS
            )
            if observed != EXPECTED_DETERMINISTIC_6DP[scenario][method]:
                raise ValueError(
                    f"Deterministic MCC mismatch: S{scenario}, {method}, {observed}"
                )
        if round(full[scenario], 6) != EXPECTED_FULL_6DP[scenario]:
            raise ValueError(f"FULL mismatch for Scenario {scenario}: {full[scenario]}")

    random_rows = read_csv(RANDOM_SCENARIO_SOURCE)
    if len(random_rows) != 15:
        raise ValueError(f"Expected 15 Random aggregate rows; found {len(random_rows)}.")
    seen: set[tuple[int, float]] = set()
    for row in random_rows:
        scenario = int(row["scenario"])
        budget = float(row["budget"])
        key = (scenario, budget)
        if scenario not in SCENARIOS or budget not in BUDGETS or key in seen:
            raise ValueError(f"Invalid or duplicate Random aggregate row: {row}")
        seen.add(key)
        if int(row["seed_count"]) != 30:
            raise ValueError(f"Random seed_count is not 30: {row}")
        if parse_seed_set(row["seed_set"]) != EXPECTED_SEEDS:
            raise ValueError(f"Random seed set is not exactly 1--30: {row}")
        if int(row["K_windows"]) != K_BY_BUDGET[budget]:
            raise ValueError(f"Random K mismatch: {row}")
        reduced[scenario]["random"][budget] = float(row["mean_MCC"])
    if seen != {(scenario, budget) for scenario in SCENARIOS for budget in BUDGETS}:
        raise ValueError("Random aggregate grid is incomplete.")
    for scenario in SCENARIOS:
        observed = tuple(reduced[scenario]["random"][budget] for budget in BUDGETS)
        expected = EXPECTED_RANDOM[scenario]
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)
               for a, b in zip(observed, expected)):
            raise ValueError(f"Random scenario-mean mismatch for Scenario {scenario}.")

    worst_rows = read_csv(RANDOM_WORST_SOURCE)
    if len(worst_rows) != 5:
        raise ValueError(f"Expected 5 seed-wise-worst rows; found {len(worst_rows)}.")
    random_worst: dict[float, float] = {}
    for row in worst_rows:
        budget = float(row["budget"])
        if budget not in BUDGETS or budget in random_worst:
            raise ValueError(f"Invalid or duplicate seed-wise-worst row: {row}")
        if int(row["seed_count"]) != 30 or parse_seed_set(row["seed_set"]) != EXPECTED_SEEDS:
            raise ValueError(f"Seed-wise-worst row does not use exactly seeds 1--30: {row}")
        if int(row["K_windows"]) != K_BY_BUDGET[budget]:
            raise ValueError(f"Seed-wise-worst K mismatch: {row}")
        random_worst[budget] = float(row["mean_seed_worst_MCC"])
    observed_worst = tuple(random_worst[budget] for budget in BUDGETS)
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)
           for a, b in zip(observed_worst, EXPECTED_RANDOM_WORST)):
        raise ValueError("Random seed-wise-worst values do not match the verified values.")

    limiting_counts = {2: 0, 3: 0}
    max_farw_gap = 0.0
    for budget in BUDGETS:
        deterministic_worst = {
            method: min(reduced[s][method][budget] for s in SCENARIOS)
            for method in DETERMINISTIC_METHODS
        }
        for method in DETERMINISTIC_METHODS:
            limiting = min(SCENARIOS, key=lambda s: reduced[s][method][budget])
            if limiting not in limiting_counts:
                raise ValueError(f"Unexpected limiting scenario: S{limiting}")
            limiting_counts[limiting] += 1
        summaries = deterministic_worst | {"random": random_worst[budget]}
        leader = max(METHOD_ORDER, key=lambda method: summaries[method])
        if leader != EXPECTED_LEADERS[budget]:
            raise ValueError(f"Leader mismatch at {budget:.0%}: {leader}")
        max_farw_gap = max(max_farw_gap, summaries[leader] - summaries["farw"])
    if limiting_counts != {2: 5, 3: 15}:
        raise ValueError(f"Deterministic limiting-scenario count mismatch: {limiting_counts}")
    if not math.isclose(max_farw_gap, 0.013335, rel_tol=0.0, abs_tol=5e-7):
        raise ValueError(f"Unexpected maximum FARW gap: {max_farw_gap}")

    return reduced, full, random_worst


def configure_typography() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "legend.fontsize": 7.9,
            "axes.linewidth": 0.80,
            "lines.linewidth": 1.60,
            "lines.markersize": 5.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="y", which="major", color="#D9DEE5", linewidth=0.52, alpha=0.75)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#50565E")
    axis.spines["bottom"].set_color("#50565E")
    axis.tick_params(direction="out", length=3.0, width=0.75, colors="#30343B")
    axis.set_xlim(8, 52)
    axis.set_xticks((10, 20, 30, 40, 50))
    axis.set_xlabel("Retention (%)")
    axis.set_ylim(*Y_LIMITS)
    axis.set_yticks(Y_TICKS)


def plot_figure(
    reduced: dict[int, dict[str, dict[float, float]]], full: dict[int, float]
) -> None:
    configure_typography()
    retention = [100.0 * budget for budget in BUDGETS]
    plotted_values = [
        reduced[scenario][method][budget]
        for scenario in SCENARIOS
        for method in METHOD_ORDER
        for budget in BUDGETS
    ] + [full[scenario] for scenario in SCENARIOS]
    if min(plotted_values) < Y_LIMITS[0] or max(plotted_values) > Y_LIMITS[1]:
        raise ValueError(
            f"Requested shared y-limits {Y_LIMITS} clip plotted MCC values "
            f"[{min(plotted_values):.6f}, {max(plotted_values):.6f}]."
        )
    figure, axes = plt.subplots(1, 3, figsize=(7.35, 3.35), sharex=True, sharey=True)
    panel_letters = {1: "a", 2: "b", 3: "c"}
    method_handles = []

    for scenario, axis in zip(SCENARIOS, axes):
        style_axis(axis)
        axis.set_title(
            f"({panel_letters[scenario]})  {SCENARIO_TITLES[scenario]}",
            loc="left",
            y=1.065,
            fontweight="bold",
            pad=0,
        )
        axis.axhline(
            full[scenario], color=FULL_STYLE["color"], linewidth=1.15,
            linestyle="-", zorder=1,
        )
        for method in METHOD_ORDER:
            style = STYLES[method]
            values = [reduced[scenario][method][budget] for budget in BUDGETS]
            line = axis.plot(
                retention,
                values,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markeredgecolor=style["color"],
                markeredgewidth=0.85,
                markerfacecolor="white" if method != "farw" else style["color"],
                zorder=4 if method == "farw" else 3,
            )[0]
            if scenario == 1:
                method_handles.append(line)

    axes[0].set_ylabel("MCC")
    full_handle = Line2D(
        [0], [0], color=FULL_STYLE["color"], linewidth=1.15,
        linestyle="-", label=FULL_STYLE["label"],
    )
    figure.legend(
        handles=method_handles + [full_handle],
        labels=[STYLES[method]["label"] for method in METHOD_ORDER]
        + [FULL_STYLE["label"]],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=False,
        handlelength=2.35,
        handletextpad=0.45,
        columnspacing=1.18,
        labelspacing=0.52,
    )
    figure.subplots_adjust(left=0.074, right=0.991, top=0.835, bottom=0.255, wspace=0.10)

    fixed_time = datetime(2026, 9, 16, 0, 0, tzinfo=timezone.utc)
    pdf_metadata = {
        "Title": "Scenario-specific MCC under condition shifts",
        "Author": "FARW study",
        "Subject": "Unified three-feature chatter classification with 30-seed Random baseline",
        "Keywords": "FARW, MCC, chatter, window reduction, condition shift",
        "Creator": "generate_scenario_mcc_30seed_random.py",
        "CreationDate": fixed_time,
        "ModDate": fixed_time,
    }
    figure.savefig(OUTPUT_PDF, format="pdf", metadata=pdf_metadata)
    figure.savefig(
        OUTPUT_PNG,
        format="png",
        dpi=600,
        metadata={"Software": "generate_scenario_mcc_30seed_random.py"},
    )
    plt.close(figure)


def main() -> None:
    reduced, full, random_worst = load_and_validate()
    plot_figure(reduced, full)
    print(f"Frozen Unified-3F source: {FROZEN_SOURCE}")
    print(f"Frozen source SHA-256: {sha256(FROZEN_SOURCE)}")
    print(f"Random scenario source: {RANDOM_SCENARIO_SOURCE}")
    print(f"Random scenario SHA-256: {sha256(RANDOM_SCENARIO_SOURCE)}")
    print(f"Random worst source: {RANDOM_WORST_SOURCE}")
    print(f"Random worst SHA-256: {sha256(RANDOM_WORST_SOURCE)}")
    print("Validation: 60 deterministic cells + 3 FULL references PASS")
    print("Validation: 15 Random means, seed_count=30, and seeds=1--30 PASS")
    print("Validation: 5 seed-wise worst means and 5 budget leaders PASS")
    print("Validation: deterministic source contains no Random rows PASS")
    print("Random seed-wise worst means: " + ", ".join(
        f"{100 * budget:.0f}%={random_worst[budget]:.15f}" for budget in BUDGETS
    ))
    print(f"Vector PDF: {OUTPUT_PDF}")
    print(f"600-dpi PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
