from helper_functions import *
from traces_plot import *
from pathlib import Path
import os


def process_folder(input_folder, pixel_spacing, pixel_window, pixel_fit_width):
    input_folder = Path(input_folder)
    tif_files = sorted([f for f in input_folder.glob("*.tif") if "_fitted" not in f.name])

    for image_path in tif_files:
        print(f"Processing {image_path.name}")
        process_file(image_path, pixel_spacing, pixel_window, pixel_fit_width)


def process_file(image_path, pixel_spacing, pixel_window, pixel_fit_width):
    traces_path = image_path.with_suffix(".traces")
    output_image = image_path.with_name(image_path.stem + "_fitted.tif")
    sliding_window = int(round(pixel_window / pixel_spacing))
    sliding_window += 1 - sliding_window % 2  # round to nearest odd number

    if not traces_path.exists():
        print(f"Skipping: No matching .traces file for {image_path.name}")
        return

    channel0, _ = load_tiff_channels(image_path)
    traces, metadata = load_and_filter_traces(traces_path)
    neurite_types = get_neurite_types(metadata)

    smoothed = [interpolate_and_smooth(t, window=sliding_window, spacing=pixel_spacing) for t in traces]
    results = [fit_trace_width(channel0, t, fit_width=pixel_fit_width, window=sliding_window) for t in smoothed]
    intensities, widths, translations, fits, normals = zip(*results)
    translated = [translate_trace(t, tr, n) for t, tr, n in zip(smoothed, translations, normals)]

    df = assemble_dataframe(smoothed, neurite_types, widths, translations, normals, intensities, fits)
    df.to_csv(image_path.with_suffix(".csv"), index=False)
    print(f"Saved all measurements to: {image_path.with_suffix('.csv')}")

    plot_traces(channel0, translated, widths, normals, save_path=output_image)


if __name__ == "__main__":
    # ---- Run analysis ---- #
    base_dir = r".\GTP_Data"
    folders = ["GradientTest",
               "DIV3_GTP_Tubulin_R1",
               "DIV3_GTP_Tubulin_R2",
               "DIV3_GTP_Tubulin_R2"]

    for folder in folders:
        full_path = os.path.join(base_dir, folder)
        process_folder(full_path, pixel_spacing=3, pixel_window=20, pixel_fit_width=50)
