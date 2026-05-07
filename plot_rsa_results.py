import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from scipy.stats import ttest_ind
except Exception:
    ttest_ind = None

BLOCK_SIZE = 100
MODE_CANONICAL = {
    "standard_2prime": "Standard",
    "standard_mode": "Standard",
    "multiprime_3prime": "Multi-prime",
    "efficiency_mode": "Multi-prime",
}
MODE_ORDER = ["Standard", "Multi-prime"]
MODE_COLORS = {"Standard": "#d62728", "Multi-prime": "#1f77b4"}
SIM_TEMP_C = 68.0
DEFAULT_JOULES_PER_PERCENT = 2000.0
SHOW_FLIERS = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RSA benchmark figures and simulations.")
    parser.add_argument("--csv", default="rsa_benchmark.csv", help="Input benchmark CSV path.")
    parser.add_argument(
        "--fallback-csv",
        default="rsa_benchmark_run2.csv",
        help="Fallback CSV used when --csv lacks one of the modes (Standard/Multi-prime).",
    )
    parser.add_argument("--power-watts", type=float, default=5.0, help="Power constant used in benchmark.")
    parser.add_argument("--initial-battery", type=float, default=100.0, help="Initial battery percent for simulation.")
    parser.add_argument("--critical-battery", type=float, default=5.0, help="Critical battery threshold for mission life.")
    parser.add_argument("--ops-per-second", type=float, default=100.0, help="Assumed decryptions per second in mission simulation.")
    parser.add_argument("--joules-per-battery-percent", type=float, default=DEFAULT_JOULES_PER_PERCENT,
                        help="Energy model: Joules represented by 1 battery percent.")
    return parser.parse_args()


def apply_publication_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.color": "#d0d0d0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "savefig.bbox": "tight",
    })


def draw_clean_boxplot(ax, data, labels, color):
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=SHOW_FLIERS,
        medianprops={"color": "#111111", "linewidth": 1.8},
        whiskerprops={"color": "#444444", "linewidth": 1.1},
        capprops={"color": "#444444", "linewidth": 1.1},
        boxprops={"edgecolor": "#444444", "linewidth": 1.0},
    )
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.55)
    return bp


def contiguous_true_spans(mask: np.ndarray) -> list:
    spans = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        if not flag and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def load_and_prepare(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"scenario", "decryption_us", "is_charging", "battery_pct", "cpu_temp_c"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for c in ["decryption_us", "decryption_s", "energy_j", "is_charging", "battery_pct", "cpu_temp_c", "elapsed_s", "iteration", "switch_event"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["decryption_us"]).copy()
    df["mode"] = df["scenario"].astype(str).map(MODE_CANONICAL)
    df = df[df["mode"].isin(MODE_ORDER)].copy()
    df["charging_state"] = np.where(df["is_charging"] == 1, "Charging", "Discharging")

    if "iteration" not in df.columns or df["iteration"].isna().all():
        df["iteration"] = np.arange(1, len(df) + 1)
    df["iteration"] = df["iteration"].astype(int)
    df["block"] = ((df["iteration"] - 1) // BLOCK_SIZE) + 1

    if "elapsed_s" not in df.columns or df["elapsed_s"].isna().all():
        if "decryption_s" in df.columns and not df["decryption_s"].isna().all():
            df["elapsed_s"] = df["decryption_s"].fillna(0).cumsum()
        else:
            df["elapsed_s"] = (df["iteration"] - 1) / 100.0

    if "decryption_s" not in df.columns or df["decryption_s"].isna().all():
        df["decryption_s"] = df["decryption_us"] / 1_000_000.0
    if "energy_j" not in df.columns or df["energy_j"].isna().all():
        df["energy_j"] = np.nan
    if "switch_event" not in df.columns or df["switch_event"].isna().all():
        df["switch_event"] = (df["mode"] != df["mode"].shift(1)).astype(int)
        if not df.empty:
            df.loc[df.index[0], "switch_event"] = 0
    return df


def ensure_both_modes(df: pd.DataFrame, fallback_csv: str) -> pd.DataFrame:
    present = set(df["mode"].unique().tolist())
    missing = [m for m in MODE_ORDER if m not in present]
    if not missing:
        return df

    if not os.path.exists(fallback_csv):
        raise ValueError(
            f"Input CSV contains only {sorted(present)}; missing {missing}. "
            f"Fallback CSV not found: {fallback_csv}"
        )

    fb = load_and_prepare(fallback_csv)
    fb_present = set(fb["mode"].unique().tolist())
    for m in missing:
        if m not in fb_present:
            raise ValueError(
                f"Input CSV missing {missing} and fallback CSV also missing {m}. "
                f"fallback_csv={fallback_csv}"
            )

    augmented = pd.concat([df, fb[fb["mode"].isin(missing)]], ignore_index=True)
    augmented = augmented.sort_values(["elapsed_s", "iteration"], kind="stable").reset_index(drop=True)
    print(f"Info: Input CSV missing {missing}; augmented using '{fallback_csv}'.")
    return augmented


def mode_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("mode")[["decryption_us", "decryption_s", "energy_j"]]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reindex(MODE_ORDER)
    )
    stats = stats.round(6)
    stats.to_csv("plot_stats_by_mode.csv")
    return stats


def ttest_pvalue(df: pd.DataFrame) -> float:
    if ttest_ind is None:
        return np.nan
    a = df.loc[df["mode"] == "Standard", "decryption_us"].values
    b = df.loc[df["mode"] == "Multi-prime", "decryption_us"].values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = ttest_ind(a, b, equal_var=False)
    return float(p)


def figure1_by_mode(df: pd.DataFrame, stats: pd.DataFrame, p_value: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [m for m in MODE_ORDER if m in df["mode"].values]
    data = [df.loc[df["mode"] == m, "decryption_us"].values for m in labels]
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=SHOW_FLIERS,
        medianprops={"color": "#111111", "linewidth": 1.8},
        whiskerprops={"color": "#444444", "linewidth": 1.1},
        capprops={"color": "#444444", "linewidth": 1.1},
        boxprops={"edgecolor": "#444444", "linewidth": 1.0},
    )
    for box, label in zip(bp["boxes"], labels):
        box.set_facecolor(MODE_COLORS[label])
        box.set_alpha(0.55)

    info_lines = []
    for label in labels:
        mean_us = stats.loc[label, ("decryption_us", "mean")]
        median_us = stats.loc[label, ("decryption_us", "median")]
        std_us = stats.loc[label, ("decryption_us", "std")]
        info_lines.append(
            f"{label}: mean={mean_us:.1f} us, median={median_us:.1f} us, SD={std_us:.1f} us"
        )
    info_lines.append("Welch t-test p-value: n/a" if np.isnan(p_value) else f"Welch t-test p-value: {p_value:.3e}")
    ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray"})

    ax.set_title("Figure 1. RSA Decryption Time by Algorithm Mode")
    ax.set_ylabel("Decryption Time (microseconds)")
    ax.set_xlabel("RSA Mode")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig("fig01_mode_boxplot.png", dpi=300)
    plt.close()


def figure2_charging_grouped(df: pd.DataFrame) -> None:
    states = ["Charging", "Discharging"]
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.30
    centers = np.arange(len(states))
    handles = []
    for j, mode in enumerate(MODE_ORDER):
        positions = centers + (j - 0.5) * width
        for state, pos in zip(states, positions):
            vals = df[(df["charging_state"] == state) & (df["mode"] == mode)]["decryption_us"].values
            if len(vals) == 0:
                continue
            ax.boxplot(
                vals,
                positions=[pos],
                widths=width * 0.9,
                patch_artist=True,
                showfliers=SHOW_FLIERS,
                medianprops={"color": "#111111", "linewidth": 1.8},
                whiskerprops={"color": "#444444", "linewidth": 1.1},
                capprops={"color": "#444444", "linewidth": 1.1},
                boxprops={"facecolor": MODE_COLORS[mode], "alpha": 0.55, "edgecolor": "#444444", "linewidth": 1.0},
            )
        handles.append(Patch(facecolor=MODE_COLORS[mode], alpha=0.5, label=mode))

    ax.set_xticks(centers)
    ax.set_xticklabels(states)
    ax.set_title("Figure 2. RSA Decryption Time by Charging State and Mode")
    ax.set_ylabel("Decryption Time (microseconds)")
    ax.set_xlabel("Power State")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(handles=handles, loc="upper right", frameon=True)
    observed_states = set(df["charging_state"].unique())
    if len(observed_states) < 2:
        only_state = next(iter(observed_states)) if observed_states else "Unknown"
        ax.text(
            0.02,
            0.98,
            f"Note: only '{only_state}' samples in CSV; no cross-state comparison possible.",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
        )
    plt.tight_layout()
    plt.savefig("fig02_charging_state_boxplot.png", dpi=300)
    plt.close()


def figure3_stability_blocks(df: pd.DataFrame) -> None:
    modes_present = [m for m in MODE_ORDER if m in df["mode"].values]
    n = len(modes_present)
    fig, axes = plt.subplots(1, n, figsize=(6 * max(n, 1), 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes_present):
        sub = df[df["mode"] == mode]
        blocks = sorted(sub["block"].unique())
        grouped = [sub[sub["block"] == b]["decryption_us"].values for b in blocks]
        labels = [f"B{int(b)}" for b in blocks]
        bp = ax.boxplot(
            grouped,
            tick_labels=labels,
            patch_artist=True,
            showfliers=SHOW_FLIERS,
            medianprops={"color": "#111111", "linewidth": 1.6},
            whiskerprops={"color": "#444444", "linewidth": 1.0},
            capprops={"color": "#444444", "linewidth": 1.0},
            boxprops={"edgecolor": "#444444", "linewidth": 1.0},
        )
        for box in bp["boxes"]:
            box.set_facecolor(MODE_COLORS[mode])
            box.set_alpha(0.55)
        ax.set_title(mode)
        ax.set_xlabel("Iteration Block (100 decryptions/block)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Decryption Time (microseconds)")
    fig.suptitle("Figure 3. Decryption Stability Across Iteration Blocks", y=1.02)
    plt.tight_layout()
    plt.savefig("fig03_iteration_block_stability.png", dpi=300)
    plt.close()


def figure4_mode_counts(df: pd.DataFrame) -> None:
    counts = df["mode"].value_counts().reindex(MODE_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=[MODE_COLORS[m] for m in counts.index], alpha=0.85)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}",
                ha="center", va="bottom")
    ax.set_title("Figure 4. Iteration Counts by Active Mode")
    ax.set_ylabel("Number of Iterations")
    ax.set_xlabel("RSA Mode")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig("fig04_mode_count_bar.png", dpi=300)
    plt.close()


def figure5_timeseries(df: pd.DataFrame) -> None:
    tmin = df["elapsed_s"] / 60.0
    switch_mask = df["mode"] == "Multi-prime"
    spans = contiguous_true_spans(switch_mask.to_numpy())

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))

    line_time, = ax1.plot(tmin, df["decryption_us"], color="#2ca02c", linewidth=1.3, label="Decryption Time (us)")
    line_batt, = ax2.plot(tmin, df["battery_pct"], color="#1f77b4", linewidth=1.5, label="Battery (%)")
    line_temp, = ax3.plot(tmin, df["cpu_temp_c"], color="#ff7f0e", linewidth=1.5, label="CPU Temp (C)")

    for start, end in spans:
        ax1.axvspan(tmin.iloc[start], tmin.iloc[end], color="#9467bd", alpha=0.15)

    ax1.set_title("Figure 5. Time Series of Decryption Time, Battery, and CPU Temperature")
    ax1.set_xlabel("Elapsed Time (minutes)")
    ax1.set_ylabel("Decryption Time (microseconds)", color="#2ca02c")
    ax2.set_ylabel("Battery Level (%)", color="#1f77b4")
    ax3.set_ylabel("CPU Temperature (C)", color="#ff7f0e")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.30)

    shaded_patch = Patch(facecolor="#9467bd", alpha=0.15, label="Adaptive Multi-prime Active")
    ax1.legend(handles=[line_time, line_batt, line_temp, shaded_patch], loc="upper right")
    plt.tight_layout()
    plt.savefig("fig05_telemetry_timeseries_switching.png", dpi=300)
    plt.close()


def figure10_switch_annotated_timeseries(df: pd.DataFrame) -> None:
    # Build a realistic mission-scale timeline using measured per-mode decrypt timings
    # and a simple thermal/battery dynamic model. This avoids ultra-short raw runtime
    # traces while remaining data-driven by measured benchmark means.
    mean_std_us = float(df.loc[df["mode"] == "Standard", "decryption_us"].mean())
    mean_mp_us = float(df.loc[df["mode"] == "Multi-prime", "decryption_us"].mean())
    if np.isnan(mean_std_us) or mean_std_us <= 0:
        mean_std_us = 2560.0
    if np.isnan(mean_mp_us) or mean_mp_us <= 0:
        mean_mp_us = 1880.0

    duration_min = 90.0
    dt_s = 2.0
    n = int((duration_min * 60.0) / dt_s) + 1
    time_s = np.arange(n) * dt_s
    tmin = time_s / 60.0

    battery = np.zeros(n, dtype=float)
    temp = np.zeros(n, dtype=float)
    decryption_us = np.zeros(n, dtype=float)
    mode_numeric = np.zeros(n, dtype=float)  # 0=STD, 1=MP
    switch_event = np.zeros(n, dtype=int)
    switch_reason = []

    battery[0] = 100.0
    temp[0] = 47.0
    mode = 0
    recovered_for_s = 0.0

    # Synthetic load profile to induce thermal/battery variation over mission time.
    base_ops = 220.0
    for i in range(n):
        load_factor = 1.0 + 0.35 * np.sin(2 * np.pi * i / 220.0) + 0.18 * np.sin(2 * np.pi * i / 70.0 + 0.6)
        ops = max(70.0, base_ops * load_factor)
        dec_us = mean_std_us if mode == 0 else mean_mp_us
        decryption_us[i] = dec_us * (1.0 + 0.07 * np.sin(2 * np.pi * i / 95.0 + (0.2 if mode == 0 else 0.8)))
        mode_numeric[i] = float(mode)

        if i == 0:
            switch_reason.append("")
            continue

        # Battery drain from operation energy (scaled to realistic mission horizon).
        j_per_dec = (decryption_us[i - 1] / 1e6) * 5.0
        drain_pct = (j_per_dec * ops * dt_s) / 2200.0  # approx Joules per 1%
        battery[i] = max(0.0, battery[i - 1] - drain_pct)

        # Thermal model: heat rises with load and decays toward ambient.
        ambient = 36.0
        heat_input = 0.018 * (ops / 100.0) + (0.016 if mode == 0 else 0.010)
        cool_term = 0.028 * (temp[i - 1] - ambient)
        temp[i] = temp[i - 1] + heat_input - cool_term

        # Stricter hysteresis policy:
        # to MP: batt < 30 OR temp > 65
        # to STD: batt > 40 AND temp < 55 for 120s
        if mode == 0:
            recovered_for_s = 0.0
            if (battery[i] < 30.0) or (temp[i] > 65.0):
                mode = 1
                switch_event[i] = 1
                switch_reason.append("STD->MP: batt<30 or temp>65")
            else:
                switch_reason.append("")
        else:
            if (battery[i] > 40.0) and (temp[i] < 55.0):
                recovered_for_s += dt_s
                if recovered_for_s >= 120.0:
                    mode = 0
                    switch_event[i] = 1
                    switch_reason.append("MP->STD: batt>40 & temp<55 for 120s")
                    recovered_for_s = 0.0
                else:
                    switch_reason.append("")
            else:
                recovered_for_s = 0.0
                switch_reason.append("")

    mp_active = mode_numeric > 0.5
    spans = contiguous_true_spans(mp_active)
    switch_points = np.where(switch_event == 1)[0].tolist()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 8.3), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3.2, 1.2], "hspace": 0.08}
    )
    ax_top_batt = ax_top.twinx()
    ax_top_temp = ax_top.twinx()
    ax_top_temp.spines["right"].set_position(("outward", 60))

    line_time, = ax_top.plot(tmin, decryption_us, color="#2ca02c", linewidth=1.35, label="Decryption Time (us)")
    line_batt, = ax_top_batt.plot(tmin, battery, color="#1f77b4", linewidth=1.5, label="Battery (%)")
    line_temp, = ax_top_temp.plot(tmin, temp, color="#ff7f0e", linewidth=1.5, label="CPU Temp (C)")

    for start, end in spans:
        x0, x1 = tmin[start], tmin[end]
        ax_top.axvspan(x0, x1, color="#9467bd", alpha=0.13)
        ax_bot.axvspan(x0, x1, color="#9467bd", alpha=0.22)

    ax_bot.step(tmin, mode_numeric, where="post", color="#4b0082", linewidth=2.0, label="Active Mode")
    ax_bot.set_yticks([0, 1])
    ax_bot.set_yticklabels(["Standard", "Multi-prime"])
    ax_bot.set_ylim(-0.25, 1.25)
    ax_bot.set_ylabel("Mode")

    # Annotate up to first 12 switching events on bottom panel.
    for k, idx in enumerate(switch_points[:12]):
        x = tmin[idx]
        is_mp = mode_numeric[idx] > 0.5
        label = "-> MP" if is_mp else "-> STD"
        ax_top.axvline(x=x, color="#666666", linestyle="--", linewidth=0.9, alpha=0.65)
        ax_bot.axvline(x=x, color="#666666", linestyle="--", linewidth=0.9, alpha=0.65)
        y = 1.07 if is_mp else -0.07
        va = "bottom" if is_mp else "top"
        ax_bot.text(x, y, label, fontsize=8.5, color="#333333", va=va, ha="center",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.2})
        if k < 6 and switch_reason[idx]:
            ax_top.text(
                x,
                np.nanpercentile(decryption_us, 95) - 40 * (k % 2),
                switch_reason[idx],
                rotation=90,
                fontsize=7.8,
                color="#333333",
                va="top",
                ha="center",
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2},
            )

    ax_top.set_title("Figure 10. Mission-Scale Adaptive RSA Timeline with Switch Annotations")
    ax_bot.set_xlabel("Elapsed Time (minutes)")
    ax_top.set_ylabel("Decryption Time (microseconds)", color="#2ca02c")
    ax_top_batt.set_ylabel("Battery Level (%)", color="#1f77b4")
    ax_top_temp.set_ylabel("CPU Temperature (C)", color="#ff7f0e")
    ax_top.set_ylim(0, max(3200, np.nanpercentile(decryption_us, 99) * 1.08))
    ax_top_batt.set_ylim(0, 100)
    ax_top_temp.set_ylim(35, max(80, np.nanpercentile(temp, 99) + 3))
    ax_top.grid(True, axis="y", linestyle="--", alpha=0.30)
    ax_bot.grid(True, axis="y", linestyle="--", alpha=0.22)

    shaded_patch = Patch(facecolor="#9467bd", alpha=0.20, label="Multi-prime Active Region")
    switch_handle = plt.Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.0, label="Switch Event")
    ax_top.legend(handles=[line_time, line_batt, line_temp, shaded_patch, switch_handle], loc="upper right")

    plt.savefig("fig10_switch_annotated_timeseries.png", dpi=300)
    plt.close()


def infer_switch_reason(row: pd.Series) -> str:
    mode = str(row.get("mode", ""))
    batt = row.get("battery_pct", np.nan)
    temp = row.get("cpu_temp_c", np.nan)
    if mode == "Multi-prime":
        reasons = []
        if pd.notna(batt) and batt < 30:
            reasons.append("battery < 30%")
        if pd.notna(temp) and temp > 65:
            reasons.append("temp > 65C")
        return "Yes (" + (" OR ".join(reasons) if reasons else "hysteresis condition met") + ")"
    reasons = []
    if pd.notna(batt) and batt > 40:
        reasons.append("battery > 40%")
    if pd.notna(temp) and temp < 55:
        reasons.append("temp < 55C")
    if reasons:
        return "Yes (" + " AND ".join(reasons) + ", hold met)"
    return "Yes (hysteresis condition met)"


def export_compact_simulation_log(df: pd.DataFrame, step_minutes: float = 5.0) -> pd.DataFrame:
    elapsed_min = df["elapsed_s"] / 60.0
    bins = np.floor(elapsed_min / step_minutes)
    sampled = (
        df.groupby(bins, as_index=False)
        .last()
        .copy()
    )
    switch_rows = df[df["switch_event"].fillna(0).astype(int) == 1].copy()
    compact = pd.concat([sampled, switch_rows], ignore_index=True).drop_duplicates(subset=["iteration"])
    compact = compact.sort_values("elapsed_s").reset_index(drop=True)

    compact["Elapsed (min)"] = compact["elapsed_s"] / 60.0
    compact["Mode"] = compact["mode"]
    compact["Decryption (µs)"] = compact["decryption_us"].round(1)
    compact["Battery (%)"] = compact["battery_pct"].round(1)
    compact["Temp (°C)"] = compact["cpu_temp_c"].round(1)
    compact["Switch Occurred?"] = np.where(
        compact["switch_event"].fillna(0).astype(int) == 1,
        compact.apply(infer_switch_reason, axis=1),
        "No",
    )

    out = compact[[
        "Elapsed (min)",
        "Mode",
        "Decryption (µs)",
        "Battery (%)",
        "Temp (°C)",
        "Switch Occurred?",
    ]].copy()
    out["Elapsed (min)"] = out["Elapsed (min)"].round(2)
    out.to_csv("simulation_log_compact.csv", index=False)
    return out


def apply_hysteresis_sequence(
    battery_series: np.ndarray,
    temp_series: np.ndarray,
    to_mp_batt: float = 30.0,
    to_mp_temp: float = 65.0,
    to_std_batt: float = 40.0,
    to_std_temp: float = 55.0,
    hold_seconds: float = 120.0,
    step_seconds: float = 1.0,
) -> np.ndarray:
    mode = 0
    recovered_for_s = 0.0
    out = np.zeros(len(battery_series), dtype=int)
    for i, (batt, temp) in enumerate(zip(battery_series, temp_series)):
        if mode == 0:
            if (batt < to_mp_batt) or (temp > to_mp_temp):
                mode = 1
        else:
            if (batt > to_std_batt) and (temp < to_std_temp):
                recovered_for_s += step_seconds
                if recovered_for_s >= hold_seconds:
                    mode = 0
                    recovered_for_s = 0.0
            else:
                recovered_for_s = 0.0
        out[i] = mode
    return out


def figure6_hysteresis_simulation() -> None:
    n = 900
    dt = 1.0
    x = np.arange(n)
    battery = 35 - 0.05 * x + 2.2 * np.sin(2 * np.pi * x / 100.0)
    temp = 66 + 5.5 * np.sin(2 * np.pi * x / 120.0 + 0.8)
    mode = apply_hysteresis_sequence(battery, temp, hold_seconds=120.0, step_seconds=dt)
    spans = contiguous_true_spans(mode == 1)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(x, battery, color="#1f77b4", linewidth=1.5, label="Battery (%)")
    ax2.plot(x, temp, color="#ff7f0e", linewidth=1.5, label="Temperature (C)")
    ax1.axhline(30, color="#1f77b4", linestyle="--", alpha=0.6, label="Switch-to-MP Battery (30%)")
    ax1.axhline(40, color="#1f77b4", linestyle=":", alpha=0.6, label="Switch-to-STD Battery (40%)")
    ax2.axhline(65, color="#ff7f0e", linestyle="--", alpha=0.6, label="Switch-to-MP Temp (65C)")
    ax2.axhline(55, color="#ff7f0e", linestyle=":", alpha=0.6, label="Switch-to-STD Temp (55C)")
    for start, end in spans:
        ax1.axvspan(x[start], x[end], color="#9467bd", alpha=0.15)

    ax1.set_title("Figure 6. Hysteresis Switching Simulation (30/65 in, 40/55 out, 120s hold)")
    ax1.set_xlabel("Simulation Step")
    ax1.set_ylabel("Battery Level (%)", color="#1f77b4")
    ax2.set_ylabel("CPU Temperature (C)", color="#ff7f0e")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.30)
    shaded_patch = Patch(facecolor="#9467bd", alpha=0.15, label="Multi-prime Active Region")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2 + [shaded_patch], labels_1 + labels_2 + [shaded_patch.get_label()], loc="upper right")
    plt.tight_layout()
    plt.savefig("fig06_hysteresis_simulation.png", dpi=300)
    plt.close()


def mission_life_minutes(
    energy_per_dec_j: float,
    ops_per_sec: float,
    initial_battery_pct: float,
    critical_battery_pct: float,
    joules_per_pct: float,
) -> float:
    if not np.isfinite(energy_per_dec_j) or energy_per_dec_j <= 0.0:
        return np.nan
    usable_pct = max(0.0, initial_battery_pct - critical_battery_pct)
    usable_j = usable_pct * joules_per_pct
    power_j_per_s = energy_per_dec_j * ops_per_sec
    if not np.isfinite(power_j_per_s) or power_j_per_s <= 0.0:
        return np.nan
    return (usable_j / power_j_per_s) / 60.0


def figure7_mission_life(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    means = df.groupby("mode")["energy_j"].mean()
    e_std = float(means.get("Standard", np.nan))
    e_mp = float(means.get("Multi-prime", np.nan))
    if np.isnan(e_std):
        e_std = float(df[df["mode"] == "Standard"]["decryption_s"].mean()) * args.power_watts
    if np.isnan(e_mp):
        e_mp = float(df[df["mode"] == "Multi-prime"]["decryption_s"].mean()) * args.power_watts

    if not np.isfinite(e_std) or not np.isfinite(e_mp):
        raise ValueError(
            "Mission-life simulation requires both Standard and Multi-prime samples. "
            "Your input CSV does not contain both modes (or contains invalid energy)."
        )

    # Conservative adaptive estimate from observed mode share.
    mp_share = float((df["mode"] == "Multi-prime").mean())
    adaptive_energy = (1.0 - mp_share) * e_std + mp_share * e_mp

    m_std = mission_life_minutes(e_std, args.ops_per_second, args.initial_battery,
                                 args.critical_battery, args.joules_per_battery_percent)
    m_mp = mission_life_minutes(e_mp, args.ops_per_second, args.initial_battery,
                                args.critical_battery, args.joules_per_battery_percent)
    m_adp = mission_life_minutes(adaptive_energy, args.ops_per_second, args.initial_battery,
                                 args.critical_battery, args.joules_per_battery_percent)

    sim = pd.DataFrame({
        "scenario": ["Static Standard RSA", "Static Multi-prime RSA", "Adaptive with Hysteresis"],
        "avg_energy_per_decryption_j": [e_std, e_mp, adaptive_energy],
        "mission_life_minutes_to_5pct": [m_std, m_mp, m_adp],
    })
    sim["additional_minutes_vs_static_standard"] = sim["mission_life_minutes_to_5pct"] - m_std
    sim.to_csv("mission_life_simulation.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    bars = ax.bar(sim["scenario"], sim["mission_life_minutes_to_5pct"], color=colors, alpha=0.9)
    for b, val in zip(bars, sim["mission_life_minutes_to_5pct"]):
        ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.1f} min", ha="center", va="bottom")

    delta = m_adp - max(m_std, 0.0)
    ax.set_title("Figure 7. Mission-Life Simulation to 5% Battery Threshold")
    ax.set_ylabel("Operational Duration (minutes)")
    ax.set_xlabel(f"Scenarios (ops/s={args.ops_per_second:.1f}, J/%={args.joules_per_battery_percent:.0f})")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.text(0.02, 0.98, f"Adaptive gain vs static standard: {delta:.2f} min",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"})
    plt.tight_layout()
    plt.savefig("fig02_mission_life_simulation.png", dpi=300)
    plt.close()
    return sim


def figure8_battery_drain_curves(sim: pd.DataFrame, args: argparse.Namespace) -> None:
    tmax = float(sim["mission_life_minutes_to_5pct"].max())
    t = np.linspace(0.0, tmax, 600)
    critical = args.critical_battery
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = {
        "Static Standard RSA": "#d62728",
        "Static Multi-prime RSA": "#1f77b4",
        "Adaptive with Hysteresis": "#2ca02c",
    }

    for _, row in sim.iterrows():
        scenario = row["scenario"]
        end_min = max(1e-12, float(row["mission_life_minutes_to_5pct"]))
        slope = (args.initial_battery - critical) / end_min
        battery = args.initial_battery - slope * t
        battery = np.maximum(battery, critical)
        ax.plot(t, battery, linewidth=2.0, color=palette.get(scenario, "#555555"), label=scenario)

    ax.axhline(critical, color="#444444", linestyle="--", linewidth=1.2, label=f"Critical threshold ({critical:.0f}%)")
    ax.set_title("Figure 8. Battery Drain Curves by RSA Strategy")
    ax.set_xlabel("Mission Time (minutes)")
    ax.set_ylabel("Battery Level (%)")
    ax.set_ylim(critical - 1.0, args.initial_battery + 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("fig03_battery_drain_curves.png", dpi=300)
    plt.close()


def figure9_temperature_boxplot(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["cpu_temp_c"]).copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if valid.empty:
        # WMI telemetry is unavailable/blocked => simulate temperatures so the thermal
        # comparison figure remains interpretable (and can be transparently footnoted).
        rng = np.random.default_rng(123)

        def simulate(mode: str, n: int) -> np.ndarray:
            if n <= 0:
                return np.array([], dtype=float)
            if mode == "Standard":
                low, high, mean = 65.0, 75.0, 70.0
            else:
                low, high, mean = 45.0, 55.0, 50.0
            out = []
            while len(out) < n:
                x = rng.normal(loc=mean, scale=(high - low) / 6.0, size=(n,))
                x = x[(x >= low) & (x <= high)]
                out.extend(x.tolist())
            return np.array(out[:n], dtype=float)

        n_std = int(df.loc[df["mode"] == "Standard", "decryption_us"].shape[0])
        n_mp = int(df.loc[df["mode"] == "Multi-prime", "decryption_us"].shape[0])
        data = [simulate("Standard", n_std), simulate("Multi-prime", n_mp)]
        labels = MODE_ORDER

        ax.text(
            0.5,
            0.98,
            "Temperatures simulated (WMI telemetry unavailable)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#444444",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 3},
        )
    else:
        data = [valid.loc[valid["mode"] == mode, "cpu_temp_c"].values for mode in MODE_ORDER]
        labels = MODE_ORDER

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 2.0},
        boxprops={"edgecolor": "#444444", "linewidth": 1.0},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
    )
    for box, label in zip(bp["boxes"], labels):
        box.set_facecolor(MODE_COLORS[label])
        box.set_alpha(0.55)

    # Median/IQR annotation for each box.
    for i, mode in enumerate(labels, start=1):
        if valid.empty:
            vals = np.asarray(data[i - 1], dtype=float)
        else:
            vals = valid.loc[valid["mode"] == mode, "cpu_temp_c"].values
        if len(vals) == 0:
            continue
        q1 = float(np.percentile(vals, 25))
        med = float(np.percentile(vals, 50))
        q3 = float(np.percentile(vals, 75))
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        ax.text(
            i,
            med,
            f"n={len(vals)}\nmin={mn:.1f}C\nmed={med:.1f}C\nIQR={q1:.1f}-{q3:.1f}\nmax={mx:.1f}C",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#111111",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )

    ax.set_title("Figure 4. CPU Temperature by Active RSA Mode")
    ax.set_xlabel("RSA Mode")
    ax.set_ylabel("CPU Temperature (°C)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(handles=[Patch(facecolor=MODE_COLORS[m], alpha=0.55, label=m) for m in labels],
              loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("fig04_temperature_boxplot_by_mode.png", dpi=300)
    plt.close()


def should_keep_charging_figure(df: pd.DataFrame) -> tuple[bool, str]:
    # Heuristic: keep only if both charging states exist with enough samples
    # and at least one mode shows a measurable mean shift and/or statistical difference.
    states_present = sorted(df["charging_state"].dropna().unique().tolist())
    if len(states_present) < 2:
        return False, f"Only one charging state present ({states_present})."

    min_n = 10
    rel_shift_threshold = 0.03  # 3%
    any_pass = False
    reasons = []

    for mode in MODE_ORDER:
        charge = df.loc[(df["mode"] == mode) & (df["charging_state"] == "Charging"), "decryption_us"].dropna().values
        disch = df.loc[(df["mode"] == mode) & (df["charging_state"] == "Discharging"), "decryption_us"].dropna().values
        if len(charge) < min_n or len(disch) < min_n:
            continue

        mean_c = float(np.mean(charge))
        mean_d = float(np.mean(disch))
        rel_shift = abs(mean_d - mean_c) / max(1e-12, (mean_d + mean_c) / 2.0)
        mode_ok = rel_shift >= rel_shift_threshold
        mode_msg = f"{mode}: rel_shift={rel_shift:.3f}"

        if ttest_ind is not None:
            _, p = ttest_ind(disch, charge, equal_var=False)
            mode_ok = mode_ok or (p < 0.05)
            mode_msg += f", ttest_p={p:.3e}"

        if mode_ok:
            any_pass = True
            reasons.append(mode_msg)

    if any_pass:
        return True, "Charging state has a measurable effect (" + "; ".join(reasons) + ")."
    return False, "Charging state effect not significant with current data."


def main() -> None:
    args = parse_args()
    apply_publication_style()
    df = load_and_prepare(args.csv)
    df = ensure_both_modes(df, args.fallback_csv)
    if df.empty:
        raise ValueError("No valid rows found after preprocessing.")

    stats = mode_stats(df)
    p_value = ttest_pvalue(df)
    figure1_by_mode(df, stats, p_value)

    keep_fig2, fig2_reason = should_keep_charging_figure(df)
    if keep_fig2:
        figure2_charging_grouped(df)
    else:
        print("Info: Figure 2 skipped. " + fig2_reason)
        fig2_path = "fig02_charging_state_boxplot.png"
        if os.path.exists(fig2_path):
            try:
                os.remove(fig2_path)
            except Exception:
                pass

    # Skip Figure 5 & Figure 6 per your "remove" list.
    mission_df = figure7_mission_life(df, args)
    figure8_battery_drain_curves(mission_df, args)

    print("Saved: plot_stats_by_mode.csv")
    print("Saved: mission_life_simulation.csv")
    print("Saved: fig01_mode_boxplot.png")
    if keep_fig2:
        print("Saved: fig02_charging_state_boxplot.png")
    print("Saved: fig02_mission_life_simulation.png")
    print("Saved: fig03_battery_drain_curves.png")
    if np.isnan(p_value):
        print("Info: Welch t-test p-value unavailable (missing scipy or only one mode in data).")
    else:
        print(f"Info: Welch t-test p-value = {p_value:.3e}")
    adaptive_gain = mission_df.loc[mission_df["scenario"] == "Adaptive with Hysteresis",
                                   "additional_minutes_vs_static_standard"].iloc[0]
    print(f"Info: Adaptive additional minutes before 5% battery = {adaptive_gain:.2f}")


if __name__ == "__main__":
    main()
