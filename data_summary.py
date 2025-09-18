import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# ---- Parameters ---- #
plot = True  # Plot averaged traces
pixel_size = 0.1018971  # in µm
distal_end = 20  # in µm; only take the last few µm of a neurite for analysis

base_dir = r".\GTP_Data"
folders = [#"GradientTest",
           "DIV3_GTP_Tubulin_R1",
           "DIV3_GTP_Tubulin_R2",
           "DIV3_GTP_Tubulin_R3"]

# ---- Summarize analysis ---- #
ratios = []
for folder in folders:
    data_folder = os.path.join(base_dir, folder)
    csv_pattern = os.path.join(data_folder, "*.csv")

    for csv_file in glob.glob(csv_pattern):
        df = pd.read_csv(csv_file)
        neuron_name = os.path.splitext(os.path.basename(csv_file))[0]
        df["neuron"] = neuron_name

        if distal_end is not None:
            def add_dist_from_end(g):
                dx = np.diff(g["new_x"].values)
                dy = np.diff(g["new_y"].values)
                step_dists = np.sqrt(dx ** 2 + dy ** 2)
                dist_from_end = np.zeros(len(g))
                dist_from_end[-2::-1] = np.cumsum(step_dists[::-1])
                g["inv_distance"] = dist_from_end * pixel_size
                return g

            # Apply function to each trace
            df = (
                df.groupby("trace_id", group_keys=False)
                .apply(add_dist_from_end)
                .reset_index(drop=True)
            )

            # Filter to only include last few microns
            df = df[df["inv_distance"] <= distal_end].copy()

            # Reset point index within each trace
            df["point_index"] = df.groupby("trace_id").cumcount()

        if plot is True:
            # Plot averaged traces for this neuron
            fig, ax = plt.subplots(figsize=(6, 4))
            for trace_id, tdf in df.groupby("trace_id"):
                color = 'tomato' if tdf["neurite"].iloc[0] == "axon" else 'royalblue'
                if distal_end is not None:
                    ax.plot(tdf["inv_distance"], tdf["intensity"], color=color, label=tdf["neurite"].iloc[0], alpha=0.7)
                else:
                    ax.plot(tdf["point_index"], tdf["intensity"], color=color, label=tdf["neurite"].iloc[0], alpha=0.7)
            ax.set_title(f"Averaged Trace Intensities – {neuron_name}", fontsize=7)
            ax.set_xlabel("Distance from distal tip (µm)", fontsize=15)
            ax.set_ylabel("Intensity (a.u.)", fontsize=15)

            # Add legend only once for unique labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best")
            plt.tight_layout()

            # Save the figure
            save_path = os.path.join(data_folder, f"{neuron_name}_averaged_traces.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)

        # Compute average intensity per neurite type in each image
        avg_intensity = df.groupby("neurite")["intensity"].median()
        if "axon" in avg_intensity and "dendrite" in avg_intensity:
            ratio = avg_intensity["axon"] / avg_intensity["dendrite"]
            ratios.append({
                "neuron": neuron_name,
                "folder": folder,
                "axon_dendrite_ratio": ratio
            })

# ---- Statistics ---- #
ratios_df = pd.DataFrame(ratios)
ratios_df.to_csv(os.path.join(base_dir, "ratios.csv"), index=False)
folder_names = sorted(ratios_df["folder"].unique())

all_ratios = ratios_df["axon_dendrite_ratio"].values
t_stat, p_val = ttest_1samp(all_ratios, popmean=1)
print(f"Combined t-test against 1: t = {t_stat:.3f}, p = {p_val:.3g}")

# Save statistic results
N = len(all_ratios)
result_str = f"t = {t_stat:.6f}\np = {p_val:.6g}\nN = {N}\n"
with open(os.path.join(base_dir, "axon_dendrite_ratio_ttest.txt"), "w") as f:
    f.write(result_str)

# ---- Plot results ---- #
fig, ax = plt.subplots(figsize=(6, 6))

ax.boxplot(
    [all_ratios],
    patch_artist=False,
    showfliers=False,
    boxprops=dict(linewidth=2),
    whiskerprops=dict(linewidth=2),
    capprops=dict(linewidth=2),
    medianprops=dict(linewidth=2, color='red')
)

# Overlay scatter points
x_jitter = np.random.normal(loc=1, scale=0.05, size=len(all_ratios))
ax.scatter(x_jitter, all_ratios, alpha=0.6, color="black", linewidth=0, zorder=3)

# Axis labels and ticks
ax.set_ylabel("Axon/Dendrite Intensity", fontsize=15)
ax.set_xticks([1])
ax.set_xticklabels(["pooled"], fontsize=12)
plt.tight_layout()

# Aesthetics
ax.set_ylabel("Axon/Dendrite intensity (a.u.)", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()

# Save figure
output_path = os.path.join("./axon_dendrite_ratio_boxplot.png")
fig.savefig(output_path, dpi=300)
plt.close(fig)
