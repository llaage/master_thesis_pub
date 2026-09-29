#plots all in just one png called all.png 

from PIL import Image
import os
import glob
import math
import re

input_dir = "/home/llaage/wetlab/purifications/260424_97p04_16GS_50A_ako/HPLC"
cols = 4
output_path = "/home/llaage/wetlab/purifications/260424_97p04_16GS_50A_ako/HPLC/all.png"  # Change to your desired output filename


def natural_key(path):
    """
    Fallback ordering: split a filename into digit / text chunks and compare
    digit chunks as integers, so "10 008.png" sorts after "4 002.png"
    instead of before it (plain string sort puts 10-19 ahead of 4).
    """
    name = os.path.basename(path)
    chunks = re.split(r"(\d+)", name)
    return [(1, int(c), "") if c.isdigit() else (0, 0, c.lower()) for c in chunks]


def trace_key(path):
    """
    Order files by the trace counter at the END of the name -- the number just
    before the extension (001, 002, ... 024).

    Names look like "v3hd_119<newline>001.png", so they carry two numbers: the
    construct id in the prefix (053, 119, 201) and the trace counter at the end.
    Sorting on all numbers left-to-right keys on the construct id first, which
    is what pushed the v3hd_053 files (023, 024) ahead of 001. Only the
    trailing counter should decide the order.

    Files with no trailing number keep natural order and go last.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)\s*$", stem)
    if m:
        return (0, int(m.group(1)), natural_key(path))
    return (1, 0, natural_key(path))


def make_image_grid(
    input_dir,
    cols,
    output_path,
    rows=None,
    background_color=(255, 255, 255, 255),
    resize_to_first=True,
):
    """
    Create an image grid from PNGs inside a folder and display it in Jupyter.

    Parameters
    ----------
    input_dir : str
        Folder containing .png files.
    cols : int
        Number of columns in the grid.
    rows : int or None
        If None, rows will be computed automatically based on the number of images.
    output_path : str or None
        If provided, the grid image will also be saved to this path.
    background_color : tuple or str
        Background color for padding (RGBA or name). Default white.
    resize_to_first : bool
        If True, resize all images to match the first image.

    Returns
    -------
    PIL.Image
        The final grid image.
    """

    # Collect PNG files, ordered by their trailing trace number
    png_paths = glob.glob(os.path.join(input_dir, "*.png"))

    # Never fold a previously written grid back into the new one
    if output_path:
        out_abs = os.path.abspath(output_path)
        png_paths = [p for p in png_paths if os.path.abspath(p) != out_abs]

    png_paths = sorted(png_paths, key=trace_key)
    if not png_paths:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    print("Trace order:")
    for i, p in enumerate(png_paths, start=1):
        print(f"  {i:>3}. " + os.path.basename(p).replace("\n", " / "))

    n_images = len(png_paths)

    # If rows not specified → compute minimal rows
    if rows is None:
        rows = math.ceil(n_images / cols)

    # Use only enough images to fill the grid
    max_images = rows * cols
    png_paths = png_paths[:max_images]

    # Load images
    images = [Image.open(p) for p in png_paths]

    # Determine uniform cell size
    if resize_to_first:
        cell_w, cell_h = images[0].size
        images = [img.resize((cell_w, cell_h), Image.LANCZOS) for img in images]
    else:
        cell_w = max(img.width for img in images)
        cell_h = max(img.height for img in images)
        padded_images = []
        for img in images:
            canvas = Image.new("RGBA", (cell_w, cell_h), background_color)
            canvas.paste(img, (0, 0))
            padded_images.append(canvas)
        images = padded_images

    # Create the grid canvas
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    grid_img = Image.new("RGBA", (grid_w, grid_h), background_color)

    # Paste images left→right, top→bottom in the sorted order
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid_img.paste(img, (c * cell_w, r * cell_h))

    if output_path:
        # ---- Optional: downscale final image to prevent Google Slides compression ----
        max_width = 3000  # choose 2000–3500 for Slides
        if grid_img.width > max_width:
            scale = max_width / grid_img.width
            new_w = int(grid_img.width * scale)
            new_h = int(grid_img.height * scale)
            grid_img = grid_img.resize((new_w, new_h), Image.LANCZOS)
            print(f"Downscaled grid to {new_w} × {new_h} px to avoid Slides compression.")

        grid_img.save(output_path)
        print(f"Saved grid image to {output_path}")

    return grid_img

make_image_grid(input_dir, cols, output_path)
