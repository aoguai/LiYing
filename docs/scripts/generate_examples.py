#!/usr/bin/env python
"""Regenerate all README/workflow example images from fixed CLI recipes.

Each recipe invokes the project CLI (src/main.py) exactly as documented in
docs/README.md, so running this script doubles as an end-to-end stability
demo. Background colors below are official presets from data/color_zh.csv.

Usage:
    python docs/scripts/generate_examples.py [--skip-existing]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON = sys.executable
MAIN = os.path.join(ROOT, 'src', 'main.py')

# Official preset background colors (data/color_zh.csv)
WHITE = '255,255,255'      # white
BLUE = '98,139,206'        # standard ID photo blue
RED = '215,69,50'          # red
DARK_BLUE = '0,71,171'     # dark blue

RMBG_2 = os.path.join('src', 'model', 'RMBG-2.0-model.onnx')

# (name, mode, CLI arguments) — each recipe mirrors the command shown in docs/README.md
# mode: 'corrected'  -> --save-corrected artifact (pre-resize corrected image) is the product
#       'direct'     -> --no-layout main output is written straight to the target
#       'sheet'      -> layout recipe, main output placeholder, sheet goes to images/<stem>_sheet.jpg
#       'pair'       -> runs the same recipe twice with -lp 0 and -lp 8, then merges both
#                       sheets into one image whose photo grid fills the whole paper
RECIPES = [
    # --- test1 pipeline: workflow header chain (workflows.html) ---
    ('test1_output_corrected.jpg', 'corrected', [
        'images/test1.jpg', '-p', 'One Inch', '--no-layout', '--save-corrected']),
    ('test1_output_retouched.jpg', 'corrected', [
        'images/test1.jpg', '-p', 'One Inch', '--no-layout',
        '--skin-retouch', '--save-corrected']),
    ('test1_output_background.jpg', 'direct', [
        'images/test1.jpg', '-p', 'One Inch', '--no-layout', '--no-resize',
        '--change-background', '-b', WHITE]),
    ('test1_output_resized.jpg', 'direct', [
        'images/test1.jpg', '-p', 'One Inch', '--no-layout',
        '--change-background', '-b', WHITE]),
    # --- layout matrix: four photos, four preset background colors ---
    ('test1_output_sheet.jpg', 'sheet', [
        'images/test1.jpg', '-p', 'One Inch', '-ps', 'Five Inch', '-sr', '3', '-sc', '3',
        '--change-background', '-b', WHITE]),
    ('test2_output_sheet.jpg', 'sheet', [
        'images/test2.jpg', '-p', 'Two Inch (ID Photo)', '-ps', 'Five Inch', '-sr', '2', '-sc', '2',
        '--change-background', '-b', BLUE, '--no-add-crop-lines']),
    # test3 uses RMBG-2.0 for fine-grained hair matting
    ('test3_output_sheet.jpg', 'sheet', [
        'images/test3.jpg', '-p', 'One Inch', '-ps', 'Six Inch', '-sr', '4', '-sc', '2',
        '--change-background', '-b', RED, '-rt', '-r', RMBG_2,
        '--no-add-crop-lines']),
    ('test4_output_sheet.jpg', 'pair', [
        'images/test4.jpg', '-p', 'One Inch', '-ps', 'Six Inch', '-sr', '4', '-sc', '2',
        '--change-background', '-b', DARK_BLUE, '-psp', '20', '--no-add-crop-lines']),
    # --- skin retouching showcase (test4 only) ---
    ('test4_output_retouched.jpg', 'corrected', [
        'images/test4.jpg', '-p', 'One Inch', '--no-layout',
        '--skin-retouch', '--save-corrected']),
]

COMPARE_HEIGHT = 1200


def run_sheet_cli(args, out_path):
    """Run one layout CLI recipe; move its "<stem>_sheet.jpg" to out_path."""
    tmp_dir = tempfile.mkdtemp(prefix='liying_examples_')
    main_out = os.path.join(tmp_dir, 'main_out.jpg')
    cmd = [PYTHON, MAIN] + args + ['-s', main_out]
    env = {**os.environ, 'CLI_LANGUAGE': 'en'}
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', env=env)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(f'CLI failed for {out_path} (exit {result.returncode})')
    src = os.path.splitext(main_out)[0] + '_sheet.jpg'
    if not os.path.exists(src):
        raise SystemExit(f'expected sheet artifact not found: {src}')
    shutil.move(src, out_path)


def compose_full_sheet(left_path, right_path, out_path):
    """Merge a top-left layout sheet and a bottom-right layout sheet: left
    half of the output comes from the top-left variant, right half from the
    bottom-right variant, so the photo grid fills the whole paper."""
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None or left.shape != right.shape:
        raise SystemExit('layout pair inputs missing or size mismatch')

    def content_x_range(img):
        # Photo grid bounds = non-paper-white columns
        mask = (np.abs(img.astype(int) - 255).sum(axis=2) > 60)
        cols = np.where(mask.any(axis=0))[0]
        return cols.min(), cols.max()

    _, lx1 = content_x_range(left)
    rx0, _ = content_x_range(right)
    seam = int(np.clip((lx1 + rx0) // 2, 1, left.shape[1] - 1))
    out = right.copy()
    out[:, :seam] = left[:, :seam]
    cv2.imwrite(out_path, out)
    print(f'[ok  ] {os.path.basename(out_path)} (seam at x={seam})')


def run_cli(out_name, mode, args, skip_existing):
    out_path = os.path.join(ROOT, 'images', out_name)
    if skip_existing and os.path.exists(out_path):
        print(f'[skip] {out_name} (exists)')
        return out_path
    tmp_dir = tempfile.mkdtemp(prefix='liying_examples_')
    main_out = os.path.join(tmp_dir, 'main_out.jpg')
    if mode == 'corrected':
        # Main output is a throwaway; the corrected/retouched artifact lands
        # next to it as "<stem>_corrected.jpg" and is renamed to the target.
        save_arg = main_out
    elif mode == 'direct':
        save_arg = out_path
    elif mode == 'pair':
        left = os.path.join(tmp_dir, 'lp0.jpg')
        right = os.path.join(tmp_dir, 'lp8.jpg')
        run_sheet_cli(args + ['-lp', '0'], left)
        run_sheet_cli(args + ['-lp', '8'], right)
        compose_full_sheet(left, right, out_path)
        if not os.path.exists(out_path):
            raise SystemExit(f'target image missing after run: {out_path}')
        return out_path
    else:  # sheet: run with a temp main output, then move "<stem>_sheet.jpg" into images/
        save_arg = main_out
    cmd = [PYTHON, MAIN] + args + ['-s', save_arg]
    print(f'[run ] {out_name}: ' + ' '.join(args))
    env = {**os.environ, 'CLI_LANGUAGE': 'en'}
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', env=env)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(f'CLI failed for {out_name} (exit {result.returncode})')
    if mode == 'corrected':
        src = os.path.splitext(main_out)[0] + '_corrected.jpg'
        if not os.path.exists(src):
            raise SystemExit(f'expected corrected artifact not found: {src}')
        shutil.move(src, out_path)
    elif mode == 'sheet':
        src = os.path.splitext(main_out)[0] + '_sheet.jpg'
        if not os.path.exists(src):
            raise SystemExit(f'expected sheet artifact not found: {src}')
        shutil.move(src, out_path)
    if not os.path.exists(out_path):
        raise SystemExit(f'target image missing after run: {out_path}')
    return out_path


def build_compare(src_path, retouched_path, out_path):
    """Side-by-side comparison strip: original (left) vs retouched (right)."""
    src = cv2.imread(src_path)
    retouched = cv2.imread(retouched_path)
    if src is None or retouched is None:
        raise SystemExit('compare inputs missing')
    scale = COMPARE_HEIGHT / src.shape[0]
    a = cv2.resize(src, (int(src.shape[1] * scale), COMPARE_HEIGHT))
    b = cv2.resize(retouched, (int(retouched.shape[1] * COMPARE_HEIGHT / retouched.shape[0]),
                               COMPARE_HEIGHT))
    cv2.imwrite(out_path, np.hstack([a, b]))
    print(f'[ok  ] {os.path.basename(out_path)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip recipes whose target image already exists')
    opts = parser.parse_args()

    products = {}
    for name, mode, args in RECIPES:
        products[name] = run_cli(name, mode, args, opts.skip_existing)

    build_compare(
        os.path.join(ROOT, 'images', 'test4.jpg'),
        os.path.join(ROOT, 'images', 'test4_output_retouched.jpg'),
        os.path.join(ROOT, 'images', 'test4_retouch_compare.jpg'),
    )

    print('\nAll examples regenerated:')
    for name, path in products.items():
        print(f'  {name}  ->  {path}')
    print('  test4_retouch_compare.jpg  ->  images/test4_retouch_compare.jpg')


if __name__ == '__main__':
    main()
