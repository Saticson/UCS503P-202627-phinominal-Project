#!/usr/bin/env python3
"""
Sukio Platform - Brand Asset Generation & Synchronization Script
================================================================
Rebrands and compiles high-fidelity UI demonstration assets (walkthrough GIFs
and control plane screenshots) for the Sukio AutoML platform, storing all outputs
in the root `assets/` directory.

Usage:
    python scripts/generate_brand_assets.py
"""

import os
import shutil
from pathlib import Path
from PIL import Image, ImageSequence, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_GIFS = ASSETS_DIR / "gifs"
ASSETS_SCREENSHOTS = ASSETS_DIR / "screenshots"

# Source assets priority: local axiomcloud repo or current docs folder
SRC_CANDIDATES = [
    Path(r"C:\Users\keshav\Documents\axiomcloud\docs"),
    ROOT_DIR / "docs",
]

def find_source_dir() -> Path:
    for candidate in SRC_CANDIDATES:
        if candidate.exists() and (candidate / "gifs").exists():
            return candidate
    raise FileNotFoundError("Could not locate source assets directory.")

def get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    font_names = (
        ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    )
    for name in font_names:
        win_path = Path("C:/Windows/Fonts") / name
        if win_path.exists():
            return ImageFont.truetype(str(win_path), size)
    return ImageFont.load_default()

def apply_sukio_brand_screenshot(im: Image.Image) -> Image.Image:
    """Rebrands a 1000x600 control plane screenshot with Sukio identity."""
    draw = ImageDraw.Draw(im)
    bg_color = (20, 20, 24)

    # 1. Cleanly erase legacy logo & branding
    draw.rectangle([(30, 20), (250, 75)], fill=bg_color)

    # 2. Draw modern purple Sukio badge
    icon_box = [(39, 29), (69, 59)]  # 30x30
    draw.rounded_rectangle(icon_box, radius=7, fill=(139, 92, 246))

    font_icon = get_font(20, bold=True)
    bbox_s = font_icon.getbbox("S")
    w_s = bbox_s[2] - bbox_s[0]
    h_s = bbox_s[3] - bbox_s[1]
    # Center 'S' inside (54, 44)
    draw.text(
        (54 - w_s // 2 - bbox_s[0], 44 - h_s // 2 - bbox_s[1]),
        "S",
        fill=(255, 255, 255),
        font=font_icon,
    )

    # 3. Draw Brand Name & Subtitle
    font_title = get_font(16, bold=True)
    font_sub = get_font(10, bold=True)
    draw.text((80, 27), "Sukio", fill=(255, 255, 255), font=font_title)
    draw.text((81, 48), "ML CONTROL PLANE", fill=(125, 211, 252), font=font_sub)

    return im

def apply_sukio_brand_gif_frame(im: Image.Image) -> Image.Image:
    """Rebrands an 800x480 GIF frame with Sukio identity."""
    draw = ImageDraw.Draw(im)
    bg_color = (20, 20, 24)

    # 1. Erase legacy logo & branding up to x=230
    draw.rectangle([(25, 18), (230, 56)], fill=bg_color)

    # 2. Draw Sukio badge (24x24)
    icon_box = [(31, 23), (55, 47)]
    draw.rounded_rectangle(icon_box, radius=5, fill=(139, 92, 246))

    font_icon = get_font(16, bold=True)
    bbox_s = font_icon.getbbox("S")
    w_s = bbox_s[2] - bbox_s[0]
    h_s = bbox_s[3] - bbox_s[1]
    draw.text(
        (43 - w_s // 2 - bbox_s[0], 35 - h_s // 2 - bbox_s[1]),
        "S",
        fill=(255, 255, 255),
        font=font_icon,
    )

    # 3. Draw Brand Name & Subtitle
    font_title = get_font(13, bold=True)
    font_sub = get_font(8, bold=True)
    draw.text((64, 21), "Sukio", fill=(255, 255, 255), font=font_title)
    draw.text((65, 38), "ML CONTROL PLANE", fill=(125, 211, 252), font=font_sub)

    return im

def process_screenshots(src_dir: Path):
    screenshots = [
        "01_dashboard_overview.png",
        "02_dataset_diagnostics.png",
        "03_automl_training.png",
        "04_inference_explainability.png",
    ]
    ASSETS_SCREENSHOTS.mkdir(parents=True, exist_ok=True)

    for name in screenshots:
        src_path = src_dir / "screenshots" / name
        dst_path = ASSETS_SCREENSHOTS / name
        print(f"Processing screenshot: {name} -> {dst_path}")
        im = Image.open(src_path).convert("RGB")
        branded_im = apply_sukio_brand_screenshot(im)
        branded_im.save(dst_path, "PNG", optimize=True)

def process_gifs(src_dir: Path):
    ASSETS_GIFS.mkdir(parents=True, exist_ok=True)

    # 1. Process 00_frontend_control_plane.gif (full UI walkthrough requiring rebranding)
    src_main_gif = src_dir / "gifs" / "00_frontend_control_plane.gif"
    dst_main_gif = ASSETS_GIFS / "00_frontend_control_plane.gif"
    print(f"Processing animated GIF walkthrough: {src_main_gif.name} -> {dst_main_gif}")

    gif = Image.open(src_main_gif)
    new_frames = []
    durations = []

    for frame in ImageSequence.Iterator(gif):
        rgb_frame = frame.convert("RGB")
        branded_frame = apply_sukio_brand_gif_frame(rgb_frame)
        p_frame = branded_frame.convert("P", palette=Image.ADAPTIVE)
        new_frames.append(p_frame)
        durations.append(frame.info.get("duration", 100))

    new_frames[0].save(
        dst_main_gif,
        save_all=True,
        append_images=new_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    # 2. Synchronize pipeline GIFs (focused simulations without top brand header)
    pipeline_gifs = [
        "01_data_profiling.gif",
        "02_automl_training.gif",
        "03_shap_inference.gif",
    ]
    for name in pipeline_gifs:
        src_pipe = src_dir / "gifs" / name
        dst_pipe = ASSETS_GIFS / name
        print(f"Syncing pipeline GIF: {name} -> {dst_pipe}")
        shutil.copy2(src_pipe, dst_pipe)

def main():
    print(f"Starting brand asset generation for Sukio...")
    src_dir = find_source_dir()
    print(f"Source assets found at: {src_dir}")
    print(f"Target assets root: {ASSETS_DIR}")

    process_screenshots(src_dir)
    process_gifs(src_dir)
    print("All assets successfully generated and synchronized in assets/ directory!")

if __name__ == "__main__":
    main()
