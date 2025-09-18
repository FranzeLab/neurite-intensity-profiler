import numpy as np
import matplotlib.pyplot as plt
import os


def plot_traces(image, traces, all_widths, all_normals, crop=None, mask=False, save_path=None,
                trace_lw=0.6, normal_lw=0.3, normal_alpha=0.5, trace_color='black', normal_color='cyan',
                export_dpi=300):
    # Image shape
    H, W = image.shape[:2]

    # Determine canvas area
    if crop is not None:
        y0, y1, x0, x1 = crop
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H, "crop outside image"
        canvas_w, canvas_h = x1 - x0, y1 - y0
        xlim = (x0, x1)
        ylim = (y1, y0)  # flip y so origin is top-left
        img_view = image[y0:y1, x0:x1]
        extent = (x0, x1, y1, y0)
    else:
        canvas_w, canvas_h = W, H
        xlim = (0, W)
        ylim = (H, 0)
        img_view = image
        extent = (0, W, H, 0)

    # Mask: overlay as svg without image
    if mask:
        dpi_svg = export_dpi
        fig = plt.figure(figsize=(canvas_w / dpi_svg, canvas_h / dpi_svg), dpi=dpi_svg)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axis('off')
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))

        # Draw normals + traces (no image)
        for trace, widths, normals in zip(traces, all_widths, all_normals):
            if trace is None or len(trace) == 0:
                continue
            if widths is not None and normals is not None and len(widths) > 0:
                for pt, w_, n in zip(trace, widths, normals):
                    start = pt - n * (w_ / 2.0)
                    end = pt + n * (w_ / 2.0)
                    ax.plot([start[0], end[0]], [start[1], end[1]],
                            color=normal_color, linewidth=normal_lw, alpha=normal_alpha)
            ax.plot(trace[:, 0], trace[:, 1],
                    color=trace_color, linewidth=trace_lw,
                    solid_capstyle='round', solid_joinstyle='round')

        if save_path:
            root, _ = os.path.splitext(save_path)
            save_svg = root + ".svg"
            fig.savefig(save_svg, format='svg', transparent=True,
                        bbox_inches=None, pad_inches=0)
            print(f"Saved clean SVG (vectors only) to: {save_svg}")
        plt.close(fig)
        return

    # Normal mode
    dpi = export_dpi
    figsize = (canvas_w / dpi, canvas_h / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    vmin, vmax = np.percentile(img_view, [0, 99.9])
    ax.imshow(img_view, cmap='gray', vmin=vmin, vmax=vmax, origin='upper', extent=extent)
    ax.axis('off')  # Remove axes
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    # Draw normals and traces
    for trace, widths, normals in zip(traces, all_widths, all_normals):
        if trace is None or len(trace) == 0:
            continue
        if widths is not None and normals is not None and len(widths) > 0:
            for pt, w_, n in zip(trace, widths, normals):
                start = pt - n * w_ / 2
                end = pt + n * w_ / 2
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color=normal_color, linewidth=normal_lw, alpha=normal_alpha)
        ax.plot(trace[:, 0], trace[:, 1], color=trace_color, linewidth=trace_lw, alpha=0.9)

    fig.subplots_adjust(0, 0, 1, 1)

    if save_path:
        _, ext = os.path.splitext(save_path)
        ext = ext.lower()
        if ext == '.svg':
            fig.savefig(save_path, format='svg', dpi=dpi,
                        bbox_inches=None, pad_inches=0)
            print(f"Saved SVG to: {save_path}")
        else:
            fig.savefig(save_path, format='tiff', dpi=dpi,
                        bbox_inches=None, pad_inches=0)
            print(f"Saved TIFF to: {save_path}\n")

    plt.show()
