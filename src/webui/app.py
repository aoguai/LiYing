import argparse
import json
import locale
import os
import re
import sys
import time
import tempfile
import zipfile
from pathlib import Path
import shutil
from functools import partial

import cv2
import numpy as np
import gradio as gr
import pandas as pd
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)

PROJECT_ROOT = project_root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TOOL_DIR = os.path.join(src_dir, 'tool')
MODEL_DIR = os.path.join(src_dir, 'model')
SAVE_IMG_DIR = os.path.join(PROJECT_ROOT, 'output')

DEFAULT_YOLOV8_PATH = os.path.join(MODEL_DIR, 'yolov8n-pose.onnx')
DEFAULT_YUNET_PATH = os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
DEFAULT_RMBG_PATH = os.path.join(MODEL_DIR, 'RMBG-1.4-model.onnx')
DEFAULT_SIZE_CONFIG = os.path.join(DATA_DIR, 'size_{}.csv')
DEFAULT_COLOR_CONFIG = os.path.join(DATA_DIR, 'color_{}.csv')

SUPPORTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

sys.path.extend([DATA_DIR, MODEL_DIR, TOOL_DIR])

from tool.agpic import ImageCompressor
from tool.ImageProcessor import ImageProcessor
from tool.PhotoSheetGenerator import PhotoSheetGenerator
from tool.PhotoRequirements import PhotoRequirements
from tool.ConfigManager import ConfigManager

def get_language():
    """Get the system language or default to English."""
    try:
        # Try to get current locale
        current_locale = locale.getlocale()[0]
        if current_locale is None:
            # If no locale is set, try to set the default locale
            locale.setlocale(locale.LC_ALL, '')
            current_locale = locale.getlocale()[0]
        
        # Extract language code from locale
        if current_locale:
            system_lang = current_locale.split('_')[0]
            return system_lang if system_lang in ['en', 'zh'] else 'en'
    except:
        return 'en'

def load_i18n_texts():
    """Load internationalization texts from JSON files."""
    i18n_dir = os.path.join(os.path.dirname(__file__), 'i18n')
    texts = {}
    for lang in ['en', 'zh']:
        with open(os.path.join(i18n_dir, f'{lang}.json'), 'r', encoding='utf-8') as f:
            texts[lang] = json.load(f)
    return texts

TEXTS = load_i18n_texts()

def t(key, language):
    """Translate a key to the specified language."""
    return TEXTS.get(language, {}).get(key, TEXTS.get('en', {}).get(key, key))

def hsl_to_rgb(h, s, l):
    """
    Converts HSL color value to RGB.
    h: hue (0-360)
    s: saturation (0-1)
    l: lightness (0-1)
    Returns: (r, g, b) tuple, with values in the range 0-255.
    """

    h = h / 360.0
    
    if s == 0:
        r = g = b = l
    else:
        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p
        
        if l < 0.5:
            q = l * (1 + s)
        else:
            q = l + s - l * s
        
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    return (min(255, max(0, int(r * 255))),
            min(255, max(0, int(g * 255))),
            min(255, max(0, int(b * 255))))

def parse_color(color_string):
    """Parse color string to RGB list. Supports Hex, RGB/RGBA, and HSL/HSLA formats."""
    if color_string is None:
        return [255, 255, 255]
    
    # Hex
    if color_string.startswith('#'):
        return [int(color_string.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
    
    # RGB/RGBA
    rgb_match = re.match(r'rgba?\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)(?:,\s*[\d.]+)?\)', color_string)
    if rgb_match:
        return [min(255, max(0, int(float(x)))) for x in rgb_match.groups()]
    
    # HSL/HSLA
    hsl_match = re.match(r'hsla?\((\d+\.?\d*),\s*(\d+\.?\d*)%,\s*(\d+\.?\d*)%(?:,\s*[\d.]+)?\)', color_string)
    if hsl_match:
        h, s, l = hsl_match.groups()
        h = float(h) % 360
        s = min(100, max(0, float(s))) / 100.0
        l = min(100, max(0, float(l))) / 100.0
        r, g, b = hsl_to_rgb(h, s, l)
        return [r, g, b]
    
    return [255, 255, 255]

def composite_bgra_on_rgb(image, rgb_list):
    """
    Composite a BGRA image onto a solid RGB background, returning BGR.
    This is used to preview/export different background colors without re-running the model.
    """
    if image is None:
        return None

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    if not (isinstance(rgb_list, (list, tuple)) and len(rgb_list) == 3):
        raise ValueError("rgb_list must be a list/tuple of three integers (R,G,B)")

    # Already BGR
    if len(image.shape) != 3 or image.shape[2] == 3:
        return image

    # Unexpected channels, fallback to first 3 channels
    if image.shape[2] != 4:
        return image[:, :, :3]

    alpha = image[:, :, 3].astype(np.float32) / 255.0
    fg = image[:, :, :3].astype(np.float32)
    bg_bgr = np.array([rgb_list[2], rgb_list[1], rgb_list[0]], dtype=np.float32)

    out = fg * alpha[..., None] + bg_bgr * (1.0 - alpha[..., None])
    return out.astype(np.uint8)

def process_image(img_path, yolov8_path, yunet_path, rmbg_path, photo_requirements, photo_type, photo_sheet_size, rgb_list, compress=False, change_background=False, rotate=False, resize=True, sheet_rows=3, sheet_cols=3, add_crop_lines=True, layout_position=4, photos_spacing=0):
    """Process the image with specified parameters."""
    processor = ImageProcessor(img_path, 
                            yolov8_model_path=yolov8_path,
                            yunet_model_path=yunet_path,
                            RMBG_model_path=rmbg_path,
                            rgb_list=rgb_list, 
                            y_b=compress)

    processor.crop_and_correct_image()
    
    # Get file size limits from CSV if enabled
    file_size_limits = {}

    corrected_image_alpha = processor.photo.image
    
    if change_background:
        # Always generate a transparent background result so the UI can
        # composite different colors without re-running the model.
        processor.change_background([255, 255, 255, 0])
        corrected_image_alpha = processor.photo.image

    if resize:
        processor.resize_image(photo_type)

    corrected_image_alpha = processor.photo.image
    corrected_image_bgr = composite_bgra_on_rgb(corrected_image_alpha, rgb_list)

    sheet_info = photo_requirements.get_resize_image_list(photo_sheet_size)
    sheet_width, sheet_height, sheet_resolution = sheet_info['width'], sheet_info['height'], sheet_info['resolution']
    generator = PhotoSheetGenerator((sheet_width, sheet_height), sheet_resolution)
    photo_sheet_cv = generator.generate_photo_sheet(corrected_image_bgr, sheet_rows, sheet_cols, rotate, add_crop_lines, layout_position, photos_spacing)

    return {
        'final_image': photo_sheet_cv,
        'corrected_image': corrected_image_bgr,
        'corrected_image_alpha': corrected_image_alpha,
        'file_size_limits': file_size_limits
    }

def save_image(image, filename, file_format='png', resolution=300):
    """
    Save the image, supporting different formats and resolutions
    
    :param image: numpy image array
    :param filename: name of the file to save
    :param file_format: file format, such as png, jpg, tif, etc., default is png
    :param resolution: image resolution (DPI), default is 300
    """

    if not filename.lower().endswith(file_format):
        filename = f"{os.path.splitext(filename)[0]}.{file_format}"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    pil_image = Image.fromarray(image)
    pil_image.info['dpi'] = (resolution, resolution)
    pil_image.save(filename, dpi=(resolution, resolution))


def create_demo(initial_language, deployment_mode):
    """Create the Gradio demo interface."""
    config_manager = ConfigManager(language=initial_language)
    config_manager.load_configs()
    photo_requirements = PhotoRequirements(language=initial_language)

    def cleanup_temp_dir(path):
        if path and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    def update_configs():
        nonlocal photo_size_configs, sheet_size_configs, color_configs, photo_size_choices, sheet_size_choices, color_choices
        photo_size_configs = config_manager.get_photo_size_configs()
        sheet_size_configs = config_manager.get_sheet_size_configs()
        color_configs = config_manager.color_config
        photo_size_choices = list(photo_size_configs.keys())
        sheet_size_choices = list(sheet_size_configs.keys())
        color_choices = [t('custom_color', config_manager.language)] + list(color_configs.keys())

    update_configs()

    photo_size_configs = config_manager.get_photo_size_configs()
    sheet_size_configs = config_manager.get_sheet_size_configs()
    color_configs = config_manager.color_config

    photo_size_choices = list(photo_size_configs.keys())
    sheet_size_choices = list(sheet_size_configs.keys())
    color_choices = [t('custom_color', initial_language)] + list(color_configs.keys())

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        language = gr.State(initial_language)
        batch_final_images = gr.State([])
        batch_corrected_images = gr.State([])
        batch_corrected_images_alpha = gr.State([])
        batch_processed_paths = gr.State([])
        batch_download_offset = gr.State(0)
        batch_temp_dir = gr.State(None, delete_callback=cleanup_temp_dir)

        title = gr.Markdown(f"# {t('title', initial_language)}")

        color_change_source = {"source": "custom"}

        current_file_format = 'png'
        current_resolution = 300

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    label=t('upload_photo', initial_language),
                    height=160
                )
                lang_dropdown = gr.Dropdown(
                    choices=[("English", "en"), ("中文", "zh")], 
                    value=initial_language, 
                    label=t('language', initial_language)
                )

                with gr.Tabs():
                    with gr.TabItem(t('key_param', initial_language)) as key_param_tab:
                        photo_type = gr.Dropdown(
                            choices=photo_size_choices,
                            label=t('photo_type', initial_language),
                            value=photo_size_choices[0] if photo_size_choices else None
                        )
                        photo_sheet_size = gr.Dropdown(
                            choices=sheet_size_choices,
                            label=t('photo_sheet_size', initial_language),
                            value=sheet_size_choices[0] if sheet_size_choices else None
                        )
                        
                        layout_position_choices = [
                            (t('layout_position_0', initial_language), 0),
                            (t('layout_position_1', initial_language), 1),
                            (t('layout_position_2', initial_language), 2),
                            (t('layout_position_3', initial_language), 3),
                            (t('layout_position_4', initial_language), 4),
                            (t('layout_position_5', initial_language), 5),
                            (t('layout_position_6', initial_language), 6),
                            (t('layout_position_7', initial_language), 7),
                            (t('layout_position_8', initial_language), 8)
                        ]
                        layout_position = gr.Dropdown(choices=layout_position_choices, value=4, label=t('layout_position', initial_language))

                        with gr.Row():
                            preset_color = gr.Dropdown(choices=color_choices, label=t('preset_color', initial_language), value=t('custom_color', initial_language))
                            background_color = gr.ColorPicker(label=t('background_color', initial_language), value="#FFFFFF")
                        layout_only = gr.Checkbox(label=t('layout_only', initial_language), value=False)
                        sheet_rows = gr.Slider(minimum=1, maximum=10, step=1, value=3, label=t('sheet_rows', initial_language))
                        sheet_cols = gr.Slider(minimum=1, maximum=10, step=1, value=3, label=t('sheet_cols', initial_language))
                        photos_spacing = gr.Slider(minimum=0, maximum=100, step=1, value=0, label=t('photos_spacing', initial_language))
                    
                    with gr.TabItem(t('advanced_settings', initial_language)) as advanced_settings_tab:
                        yolov8_path = gr.Textbox(label=t('yolov8_path', initial_language), value=DEFAULT_YOLOV8_PATH)
                        yunet_path = gr.Textbox(label=t('yunet_path', initial_language), value=DEFAULT_YUNET_PATH)
                        rmbg_path = gr.Textbox(label=t('rmbg_path', initial_language), value=DEFAULT_RMBG_PATH)
                        size_config = gr.Textbox(label=t('size_config', initial_language), value=DEFAULT_SIZE_CONFIG.format(initial_language))
                        color_config = gr.Textbox(label=t('color_config', initial_language), value=DEFAULT_COLOR_CONFIG.format(initial_language))
                        compress = gr.Checkbox(label=t('compress', initial_language), value=True)
                        change_background = gr.Checkbox(label=t('change_background', initial_language), value=True)
                        rotate = gr.Checkbox(label=t('rotate', initial_language), value=False)
                        resize = gr.Checkbox(label=t('resize', initial_language), value=True)
                        add_crop_lines = gr.Checkbox(label=t('add_crop_lines', initial_language), value=True)
                        
                        # Add file size limit control items
                        with gr.Row():
                            use_csv_size = gr.Checkbox(label=t('use_csv_size', initial_language) if 'use_csv_size' in TEXTS[initial_language] else 'Use size limits from CSV', value=True)
                        
                        # New Radio buttons for selecting size input type
                        size_option_choices = [
                            (t('target_size_radio', initial_language), "target"), 
                            (t('size_range_radio', initial_language), "range")
                        ]
                        size_option_type = gr.Radio(
                            choices=size_option_choices,
                            label=t('size_input_option', initial_language),
                            value="target",  # Default selection when it becomes visible
                            visible=False    # Initially hidden
                        )
                        
                        with gr.Row():
                            target_size = gr.Number(label=t('target_size', initial_language) if 'target_size' in TEXTS[initial_language] else 'Target file size (KB)', precision=0, visible=False)
                        
                        with gr.Row():
                            size_range_min = gr.Number(label=t('size_range_min', initial_language) if 'size_range_min' in TEXTS[initial_language] else 'Min file size (KB)', precision=0, visible=False)
                            size_range_max = gr.Number(label=t('size_range_max', initial_language) if 'size_range_max' in TEXTS[initial_language] else 'Max file size (KB)', precision=0, visible=False)
                        
                        confirm_advanced_settings = gr.Button(t('confirm_settings', initial_language))

                    with gr.TabItem(t('config_management', initial_language)) as config_management_tab:
                        with gr.Tabs():
                            with gr.TabItem(t('size_config', initial_language)) as size_config_tab:
                                size_df = gr.Dataframe(
                                    value=pd.DataFrame(
                                        [
                                            [name] + list(config.values())
                                            for name, config in config_manager.size_config.items()
                                        ],
                                        columns=['Name'] + (list(next(iter(config_manager.size_config.values())).keys()) if config_manager.size_config else [])
                                    ),
                                    interactive=True,
                                    label=t('size_config_table', initial_language)
                                )
                                with gr.Row():
                                    add_size_btn = gr.Button(t('add_size', initial_language))
                                    update_size_btn = gr.Button(t('save_size', initial_language))

                            with gr.TabItem(t('color_config', initial_language)) as color_config_tab:
                                color_df = gr.Dataframe(
                                    value=pd.DataFrame(
                                        [
                                            [name] + list(config.values())
                                            for name, config in config_manager.color_config.items()
                                        ],
                                        columns=['Name', 'R', 'G', 'B', 'Notes']
                                    ),
                                    interactive=True,
                                    label=t('color_config_table', initial_language)
                                )
                                with gr.Row():
                                    add_color_btn = gr.Button(t('add_color', initial_language))
                                    update_color_btn = gr.Button(t('save_color', initial_language))

                        config_notification = gr.Textbox(label=t('config_notification', initial_language))

                process_btn = gr.Button(t('process_btn', initial_language))

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem(t('result', initial_language)) as result_tab:
                        output_image = gr.Image(label=t('final_image', initial_language), height=800)
                        download_final_file = gr.File(label=t('download_image', initial_language), visible=True, height=160)
                        with gr.Row():
                            save_final_btn = gr.Button(
                                t('save_image', initial_language),
                                visible=(deployment_mode == 'local')
                            )
                            save_final_path = gr.Textbox(
                                label=t('save_path', initial_language),
                                value=SAVE_IMG_DIR,
                                visible=(deployment_mode == 'local')
                            )
                    with gr.TabItem(t('corrected_image', initial_language)) as corrected_image_tab:
                        corrected_output = gr.Image(label=t('corrected_image', initial_language), height=800)
                        download_corrected_file = gr.File(label=t('download_image', initial_language), visible=True, height=160)
                        with gr.Row():
                            save_corrected_btn = gr.Button(
                                t('save_corrected', initial_language),
                                visible=(deployment_mode == 'local')
                            )
                            save_corrected_path = gr.Textbox(
                                label=t('save_path', initial_language),
                                value=SAVE_IMG_DIR,
                                visible=(deployment_mode == 'local')
                            )
                notification = gr.Textbox(label=t('notification', initial_language))

        def process_and_display(input_files, yolov8_path, yunet_path, rmbg_path, size_config, color_config, photo_type,
                                        photo_sheet_size, background_color, compress, change_background, rotate, resize,
                                        sheet_rows, sheet_cols, layout_only, add_crop_lines, layout_position, photos_spacing):
            """Process and display image(s) with given parameters (supports batch)."""
            # Update the configuration file path of ConfigManager
            config_manager.size_file = size_config
            config_manager.color_file = color_config
            config_manager.load_configs()
            update_configs()

            if input_files is None:
                return None

            if isinstance(input_files, str):
                input_paths = [input_files]
            else:
                input_paths = list(input_files)

            input_paths = [
                p for p in input_paths
                if p and Path(p).suffix.lower() in SUPPORTED_IMAGE_EXTS
            ]
            if not input_paths:
                return None

            rgb_list = parse_color(background_color)
            sheet_info = photo_requirements.get_resize_image_list(photo_sheet_size)
            file_format = sheet_info.get('file_format', 'png').lower()
            if file_format == 'jpg':
                file_format = 'jpeg'
            resolution = sheet_info.get('resolution', 300)

            final_images_rgb = []
            corrected_images_rgb = []
            corrected_images_alpha = []
            processed_paths = []
            errors = []

            for img_path in input_paths:
                try:
                    result = process_image(
                        img_path,
                        yolov8_path,
                        yunet_path,
                        rmbg_path,
                        photo_requirements,
                        photo_type=photo_type,
                        photo_sheet_size=photo_sheet_size,
                        rgb_list=rgb_list,
                        compress=compress,
                        change_background=change_background and not layout_only,
                        rotate=rotate,
                        resize=resize,
                        sheet_rows=sheet_rows,
                        sheet_cols=sheet_cols,
                        add_crop_lines=add_crop_lines,
                        layout_position=layout_position,
                        photos_spacing=photos_spacing
                    )
                except Exception as e:
                    errors.append(f"{Path(img_path).name}: {str(e)}")
                    continue

                final_images_rgb.append(cv2.cvtColor(result['final_image'], cv2.COLOR_BGR2RGB))
                corrected_images_rgb.append(cv2.cvtColor(result['corrected_image'], cv2.COLOR_BGR2RGB))
                corrected_images_alpha.append(result.get('corrected_image_alpha'))
                processed_paths.append(img_path)

            if errors:
                preview_errors = errors[:3]
                suffix = "" if len(errors) <= 3 else f" (+{len(errors) - 3})"
                gr.Warning("Failed to process some images: " + "; ".join(preview_errors) + suffix)

            if not processed_paths:
                return None

            return final_images_rgb, corrected_images_rgb, corrected_images_alpha, file_format, resolution, processed_paths

        def process_and_display_wrapper(input_files, yolov8_path, yunet_path, rmbg_path, size_config, color_config,
                                                photo_type, photo_sheet_size, background_color, compress, change_background,
                                                rotate, resize, sheet_rows, sheet_cols, layout_only, add_crop_lines,
                                                target_size, size_range_min, size_range_max, use_csv_size, layout_position, photos_spacing,
                                                lang, previous_temp_dir):
            """Wrapper for process_and_display that also prepares download files (supports batch)."""
            nonlocal current_file_format, current_resolution

            if previous_temp_dir and os.path.exists(previous_temp_dir):
                shutil.rmtree(previous_temp_dir, ignore_errors=True)

            result = process_and_display(
                input_files, yolov8_path, yunet_path, rmbg_path, size_config, color_config,
                photo_type, photo_sheet_size, background_color, compress, change_background,
                rotate, resize, sheet_rows, sheet_cols, layout_only, add_crop_lines, layout_position, photos_spacing
            )

            if not result:
                return None, None, None, None, [], [], [], [], 0, None

            final_images, corrected_images, corrected_images_alpha, file_format, resolution, processed_paths = result
            current_file_format = file_format if file_format else 'png'
            current_resolution = resolution if resolution else 300

            def sanitize_filename(name: str) -> str:
                return re.sub(r'[<>:"/\\|?*]', '_', name)

            # Build file size limits for corrected image downloads (optional)
            file_size_limits = {}
            if compress:
                if use_csv_size:
                    file_size_limits = photo_requirements.get_file_size_limits(photo_type)
                else:
                    validated_target_size = None
                    validated_size_range = None

                    if target_size is not None and target_size > 0:
                        validated_target_size = int(target_size)

                    if size_range_min is not None and size_range_max is not None and size_range_min > 0 and size_range_max > 0:
                        if size_range_min < size_range_max:
                            validated_size_range = (int(size_range_min), int(size_range_max))
                        else:
                            gr.Warning(t('size_range_error', lang) if 'size_range_error' in TEXTS[lang] else "Min size must be less than max size")

                    if validated_target_size and validated_size_range:
                        gr.Warning(t('size_params_conflict', lang) if 'size_params_conflict' in TEXTS[lang] else "Both target size and size range provided. Using target size.")
                        validated_size_range = None

                    if validated_target_size:
                        file_size_limits['target_size'] = validated_target_size
                    elif validated_size_range:
                        file_size_limits['size_range'] = validated_size_range

            output_dir = tempfile.mkdtemp(prefix="liying_webui_")

            try:
                final_paths = []
                corrected_paths = []

                for idx, img_path in enumerate(processed_paths):
                    stem = sanitize_filename(Path(img_path).stem) or f"image_{idx + 1}"

                    final_path = os.path.join(output_dir, f"{stem}_sheet.{current_file_format}")
                    save_image(final_images[idx], final_path, current_file_format, current_resolution)
                    final_paths.append(final_path)

                    corrected_path = os.path.join(output_dir, f"{stem}_corrected.{current_file_format}")
                    if compress and file_size_limits:
                        tmp_path = os.path.join(output_dir, f"__tmp_{stem}_corrected.{current_file_format}")
                        save_image(corrected_images[idx], tmp_path, current_file_format, current_resolution)
                        ImageCompressor.compress_image(
                            fp=Path(tmp_path),
                            output=Path(corrected_path),
                            force=True,
                            **file_size_limits
                        )
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    else:
                        save_image(corrected_images[idx], corrected_path, current_file_format, current_resolution)
                    corrected_paths.append(corrected_path)

                if len(processed_paths) > 1:
                    ts = int(time.time())
                    final_zip_path = os.path.join(output_dir, f"batch_final_{ts}.zip")
                    corrected_zip_path = os.path.join(output_dir, f"batch_corrected_{ts}.zip")

                    with zipfile.ZipFile(final_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                        for fp in final_paths:
                            zf.write(fp, arcname=os.path.basename(fp))

                    with zipfile.ZipFile(corrected_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                        for fp in corrected_paths:
                            zf.write(fp, arcname=os.path.basename(fp))

                    download_final_value = [final_zip_path] + final_paths
                    download_corrected_value = [corrected_zip_path] + corrected_paths
                    offset = 1
                else:
                    download_final_value = final_paths[0]
                    download_corrected_value = corrected_paths[0]
                    offset = 0

                return (
                    final_images[0],
                    corrected_images[0],
                    download_final_value,
                    download_corrected_value,
                    final_images,
                    corrected_images,
                    corrected_images_alpha,
                    processed_paths,
                    offset,
                    output_dir,
                )
            except Exception:
                shutil.rmtree(output_dir, ignore_errors=True)
                raise

        def update_background_preview(background_color, photo_type, photo_sheet_size, compress, use_csv_size,
                                      target_size, size_range_min, size_range_max, change_background, layout_only,
                                      rotate, sheet_rows, sheet_cols, add_crop_lines, layout_position, photos_spacing,
                                      lang, corrected_images_alpha, processed_paths, final_images, corrected_images,
                                      offset, previous_temp_dir):
            """
            Update preview/download outputs when only the background color changes.
            This avoids re-running the background removal model by reusing cached alpha images.
            """
            nonlocal current_file_format, current_resolution

            effective_change_background = change_background and not layout_only
            if not effective_change_background:
                return gr.update(), gr.update(), gr.update(), gr.update(), final_images, corrected_images, corrected_images_alpha, processed_paths, offset, previous_temp_dir

            if not corrected_images_alpha or not processed_paths:
                return gr.update(), gr.update(), gr.update(), gr.update(), final_images, corrected_images, corrected_images_alpha, processed_paths, offset, previous_temp_dir

            if previous_temp_dir and os.path.exists(previous_temp_dir):
                shutil.rmtree(previous_temp_dir, ignore_errors=True)

            rgb_list = parse_color(background_color)

            sheet_info = photo_requirements.get_resize_image_list(photo_sheet_size)
            file_format = sheet_info.get('file_format', 'png').lower()
            if file_format == 'jpg':
                file_format = 'jpeg'
            resolution = sheet_info.get('resolution', 300)

            current_file_format = file_format if file_format else 'png'
            current_resolution = resolution if resolution else 300

            # Rebuild corrected + sheet images (RGB for display)
            sheet_width, sheet_height = sheet_info['width'], sheet_info['height']
            generator = PhotoSheetGenerator((sheet_width, sheet_height), current_resolution)

            final_images_rgb = []
            corrected_images_rgb = []

            for idx, img in enumerate(corrected_images_alpha):
                corrected_bgr = composite_bgra_on_rgb(img, rgb_list)
                corrected_images_rgb.append(cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB))

                sheet_bgr = generator.generate_photo_sheet(
                    corrected_bgr,
                    sheet_rows,
                    sheet_cols,
                    rotate,
                    add_crop_lines,
                    layout_position,
                    photos_spacing
                )
                final_images_rgb.append(cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2RGB))

            # Build file size limits for corrected image downloads (optional)
            file_size_limits = {}
            if compress:
                if use_csv_size:
                    file_size_limits = photo_requirements.get_file_size_limits(photo_type)
                else:
                    validated_target_size = None
                    validated_size_range = None

                    if target_size is not None and target_size > 0:
                        validated_target_size = int(target_size)

                    if size_range_min is not None and size_range_max is not None and size_range_min > 0 and size_range_max > 0:
                        if size_range_min < size_range_max:
                            validated_size_range = (int(size_range_min), int(size_range_max))
                        else:
                            gr.Warning(t('size_range_error', lang) if 'size_range_error' in TEXTS[lang] else "Min size must be less than max size")

                    if validated_target_size and validated_size_range:
                        gr.Warning(t('size_params_conflict', lang) if 'size_params_conflict' in TEXTS[lang] else "Both target size and size range provided. Using target size.")
                        validated_size_range = None

                    if validated_target_size:
                        file_size_limits['target_size'] = validated_target_size
                    elif validated_size_range:
                        file_size_limits['size_range'] = validated_size_range

            def sanitize_filename(name: str) -> str:
                return re.sub(r'[<>:"/\\|?*]', '_', name)

            output_dir = tempfile.mkdtemp(prefix="liying_webui_")
            try:
                final_paths = []
                corrected_paths = []

                for idx, img_path in enumerate(processed_paths):
                    stem = sanitize_filename(Path(img_path).stem) or f"image_{idx + 1}"

                    final_path = os.path.join(output_dir, f"{stem}_sheet.{current_file_format}")
                    save_image(final_images_rgb[idx], final_path, current_file_format, current_resolution)
                    final_paths.append(final_path)

                    corrected_path = os.path.join(output_dir, f"{stem}_corrected.{current_file_format}")
                    if compress and file_size_limits:
                        tmp_path = os.path.join(output_dir, f"__tmp_{stem}_corrected.{current_file_format}")
                        save_image(corrected_images_rgb[idx], tmp_path, current_file_format, current_resolution)
                        ImageCompressor.compress_image(
                            fp=Path(tmp_path),
                            output=Path(corrected_path),
                            force=True,
                            **file_size_limits
                        )
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    else:
                        save_image(corrected_images_rgb[idx], corrected_path, current_file_format, current_resolution)
                    corrected_paths.append(corrected_path)

                if len(processed_paths) > 1:
                    ts = int(time.time())
                    final_zip_path = os.path.join(output_dir, f"batch_final_{ts}.zip")
                    corrected_zip_path = os.path.join(output_dir, f"batch_corrected_{ts}.zip")

                    with zipfile.ZipFile(final_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                        for fp in final_paths:
                            zf.write(fp, arcname=os.path.basename(fp))

                    with zipfile.ZipFile(corrected_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                        for fp in corrected_paths:
                            zf.write(fp, arcname=os.path.basename(fp))

                    download_final_value = [final_zip_path] + final_paths
                    download_corrected_value = [corrected_zip_path] + corrected_paths
                    offset = 1
                else:
                    download_final_value = final_paths[0]
                    download_corrected_value = corrected_paths[0]
                    offset = 0

                return (
                    final_images_rgb[0],
                    corrected_images_rgb[0],
                    download_final_value,
                    download_corrected_value,
                    final_images_rgb,
                    corrected_images_rgb,
                    corrected_images_alpha,
                    processed_paths,
                    offset,
                    output_dir,
                )
            except Exception:
                shutil.rmtree(output_dir, ignore_errors=True)
                raise

        def save_image_handler(image, path, lang, photo_type, photo_sheet_size, background_color, compress, use_csv_size, target_size, size_range_min, size_range_max, is_corrected, is_download_mode=False):
            nonlocal current_file_format, current_resolution, deployment_mode
            if image is None:
                return t('no_image_to_save', lang) if not is_download_mode else None
            
            if deployment_mode == 'server' and not is_download_mode:
                return t('server_save_disabled', lang)

            # Download mode uses temporary files
            if is_download_mode:
                temp_dir = tempfile.mkdtemp()
                filename = f"{photo_sheet_size}_{photo_type}_{str(background_color)}_{int(time.time())}.{current_file_format}"
                full_path = os.path.join(temp_dir, filename)
            else:
                if not path.strip():
                    path = SAVE_IMG_DIR

                path = os.path.normpath(path)

                if os.path.exists(path):
                    is_dir = os.path.isdir(path)
                else:
                    file_ext = os.path.splitext(path)[1]
                    is_dir = file_ext == ''

                if is_dir:
                    os.makedirs(path, exist_ok=True)
                    filename = f"{photo_sheet_size}_{photo_type}_{str(background_color)}_{int(time.time())}.{current_file_format}"
                    full_path = os.path.join(path, filename)
                else:
                    dir_name = os.path.dirname(path)
                    base_name, ext = os.path.splitext(os.path.basename(path))
                    base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
                    filename = f"{base_name}.{current_file_format}"

                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)

                    full_path = os.path.join(dir_name, filename) if dir_name else filename

            file_size_limits = {}
            if is_corrected and compress:
                if use_csv_size:
                    file_size_limits = photo_requirements.get_file_size_limits(photo_type)
                else:
                    validated_target_size = None
                    validated_size_range = None
                    if target_size is not None and target_size > 0:
                        validated_target_size = int(target_size)
                    if size_range_min is not None and size_range_max is not None and size_range_min > 0 and size_range_max > 0:
                        if size_range_min < size_range_max:
                            validated_size_range = (int(size_range_min), int(size_range_max))
                        else:
                             gr.Warning(t('size_range_error', lang) if 'size_range_error' in TEXTS[lang] else "Min size must be less than max size")
                    
                    if validated_target_size and validated_size_range:
                        gr.Warning(t('size_params_conflict', lang) if 'size_params_conflict' in TEXTS[lang] else "Both target size and size range provided. Using target size.")
                        validated_size_range = None
                    
                    if validated_target_size:
                        file_size_limits['target_size'] = validated_target_size
                    elif validated_size_range:
                        file_size_limits['size_range'] = validated_size_range

            temp_dir = None
            try:
                if is_corrected and compress and file_size_limits:
                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, f"temp_image.{current_file_format}")
                    save_image(image, temp_file_path, current_file_format, current_resolution)
                    
                    ImageCompressor.compress_image(
                        fp=Path(temp_file_path),
                        output=Path(full_path),
                        force=True,
                        **file_size_limits
                    )
                else:
                    save_image(image, full_path, current_file_format, current_resolution)
                
                if is_download_mode:
                    return full_path
                else:
                    return t('image_saved_success', lang).format(path=full_path)
            except Exception as e:
                if is_download_mode:
                    return None
                else:
                    return t('image_save_error', lang).format(error=str(e))
            finally:
                if temp_dir and os.path.exists(temp_dir) and not is_download_mode:
                    shutil.rmtree(temp_dir)

        def update_language(lang):
            """Update UI language and reload configs."""
            nonlocal config_manager, photo_requirements
            config_manager.switch_language(lang)
            photo_requirements.switch_language(lang)
            update_configs()
            
            new_photo_size_configs = config_manager.get_photo_size_configs()
            new_sheet_size_configs = config_manager.get_sheet_size_configs()
            color_configs = config_manager.color_config
            
            new_photo_size_choices = list(new_photo_size_configs.keys())
            new_sheet_size_choices = list(new_sheet_size_configs.keys())
            new_color_choices = [t('custom_color', lang)] + list(color_configs.keys())

            new_layout_position_choices = [
                (t('layout_position_0', lang), 0),
                (t('layout_position_1', lang), 1),
                (t('layout_position_2', lang), 2),
                (t('layout_position_3', lang), 3),
                (t('layout_position_4', lang), 4),
                (t('layout_position_5', lang), 5),
                (t('layout_position_6', lang), 6),
                (t('layout_position_7', lang), 7),
                (t('layout_position_8', lang), 8)
            ]

            # The dictionary of updates to be returned
            updates = {title: gr.update(value=f"# {t('title', lang)}"),
                       input_image: gr.update(label=t('upload_photo', lang)),
                       lang_dropdown: gr.update(label=t('language', lang)),
                       photo_type: gr.update(choices=new_photo_size_choices, label=t('photo_type', lang),
                                             value=new_photo_size_choices[0] if new_photo_size_choices else None),
                       photo_sheet_size: gr.update(choices=new_sheet_size_choices, label=t('photo_sheet_size', lang),
                                                   value=new_sheet_size_choices[0] if new_sheet_size_choices else None),
                       preset_color: gr.update(choices=new_color_choices, label=t('preset_color', lang)),
                       background_color: gr.update(label=t('background_color', lang)),
                       layout_only: gr.update(label=t('layout_only', lang)),
                       sheet_rows: gr.update(label=t('sheet_rows', lang)),
                       sheet_cols: gr.update(label=t('sheet_cols', lang)),
                       photos_spacing: gr.update(label=t('photos_spacing', lang)),
                       yolov8_path: gr.update(label=t('yolov8_path', lang)),
                       yunet_path: gr.update(label=t('yunet_path', lang)),
                       rmbg_path: gr.update(label=t('rmbg_path', lang)),
                       size_config: gr.update(label=t('size_config', lang), value=DEFAULT_SIZE_CONFIG.format(lang)),
                       color_config: gr.update(label=t('color_config', lang), value=DEFAULT_COLOR_CONFIG.format(lang)),
                       compress: gr.update(label=t('compress', lang)),
                       change_background: gr.update(label=t('change_background', lang)),
                       rotate: gr.update(label=t('rotate', lang)), resize: gr.update(label=t('resize', lang)),
                       add_crop_lines: gr.update(label=t('add_crop_lines', lang)),
                       use_csv_size: gr.update(label=t('use_csv_size', lang)),
                       target_size: gr.update(label=t('target_size', lang)),
                       size_range_min: gr.update(label=t('size_range_min', lang)),
                       size_range_max: gr.update(label=t('size_range_max', lang)),
                       process_btn: gr.update(value=t('process_btn', lang)),
                       output_image: gr.update(label=t('final_image', lang)),
                       corrected_output: gr.update(label=t('corrected_image', lang)),
                       download_final_file: gr.update(label=t('download_image', lang)),
                       download_corrected_file: gr.update(label=t('download_image', lang)),
                       save_final_btn: gr.update(value=t('save_image', lang)),
                       save_final_path: gr.update(label=t('save_path', lang)),
                       save_corrected_btn: gr.update(value=t('save_corrected', lang)),
                       save_corrected_path: gr.update(label=t('save_path', lang)),
                       notification: gr.update(label=t('notification', lang)),
                       key_param_tab: gr.update(label=t('key_param', lang)),
                       advanced_settings_tab: gr.update(label=t('advanced_settings', lang)),
                       config_management_tab: gr.update(label=t('config_management', lang)),
                       size_config_tab: gr.update(label=t('size_config', lang)),
                       color_config_tab: gr.update(label=t('color_config', lang)),
                       confirm_advanced_settings: gr.update(value=t('confirm_settings', lang)),
                       result_tab: gr.update(label=t('result', lang)),
                       corrected_image_tab: gr.update(label=t('corrected_image', lang)), size_df: gr.update(
                    value=pd.DataFrame(
                        [[name] + list(config.values()) for name, config in config_manager.size_config.items()],
                        columns=['Name'] + (list(next(
                            iter(config_manager.size_config.values())).keys()) if config_manager.size_config else [])
                    ),
                    label=t('size_config_table', lang)
                ), color_df: gr.update(
                    value=pd.DataFrame(
                        [[name] + list(config.values()) for name, config in config_manager.color_config.items()],
                        columns=['Name', 'R', 'G', 'B', 'Notes']
                    ),
                    label=t('color_config_table', lang)
                ), add_size_btn: gr.update(value=t('add_size', lang)),
                       update_size_btn: gr.update(value=t('save_size', lang)),
                       add_color_btn: gr.update(value=t('add_color', lang)),
                       update_color_btn: gr.update(value=t('save_color', lang)),
                       config_notification: gr.update(label=t('config_notification', lang)),
                       size_option_type: gr.update(
                            label=t('size_input_option', lang),
                            choices=[(t('target_size_radio', lang), "target"), (t('size_range_radio', lang), "range")]
                        ), layout_position: gr.update(choices=new_layout_position_choices, value=4,
                                                      label=t('layout_position', lang)), language: lang}
            
            return updates

        def confirm_advanced_settings_fn(yolov8_path, yunet_path, rmbg_path, size_config, color_config):
            config_manager.size_file = size_config
            config_manager.color_file = color_config
            config_manager.load_configs()
            update_configs()
            return {
                size_df: gr.update(value=pd.DataFrame(
                    [[name] + list(config.values()) for name, config in config_manager.size_config.items()],
                    columns=['Name'] + list(next(iter(config_manager.size_config.values())).keys())
                )),
                color_df: gr.update(value=pd.DataFrame(
                    [[name] + list(config.values()) for name, config in config_manager.color_config.items()],
                    columns=['Name', 'R', 'G', 'B', 'Notes']
                )),
                photo_type: gr.update(choices=photo_size_choices),
                photo_sheet_size: gr.update(choices=sheet_size_choices),
                preset_color: gr.update(choices=color_choices),
            }

        def update_background_color(preset, lang):
            """Update background color based on preset selection."""
            custom_color = t('custom_color', lang)
            if preset == custom_color:
                color_change_source["source"] = "custom"
                return gr.update()
            
            if preset in color_configs:
                color = color_configs[preset]
                hex_color = f"#{color['R']:02x}{color['G']:02x}{color['B']:02x}"
                color_change_source["source"] = "preset"
                return gr.update(value=hex_color)
            
            color_change_source["source"] = "custom"
            return gr.update(value="#FFFFFF")

        def update_preset_color(color, lang):
            """Update preset color dropdown based on color picker changes."""
            if color_change_source["source"] == "preset":
                color_change_source["source"] = "custom"
                return gr.update()
            
            # Check if the currently selected color matches any preset color
            hex_color = color.upper() if color else "#FFFFFF"
            for preset_name, preset_color in color_configs.items():
                preset_hex = f"#{preset_color['R']:02x}{preset_color['G']:02x}{preset_color['B']:02x}".upper()
                if hex_color == preset_hex:
                    return gr.update(value=preset_name)
            
            custom_color = t('custom_color', lang)
            return gr.update(value=custom_color)

        def add_size_config(df):
            """Add a new empty row to the size configuration table."""
            new_row = pd.DataFrame([['' for _ in df.columns]], columns=df.columns)
            updated_df = pd.concat([df, new_row], ignore_index=True)
            return updated_df, t('size_config_row_added', config_manager.language)

        def update_size_config(df):
            """Save all changes made to the size configuration table and remove empty rows."""
            updated_config = {}
            for _, row in df.iterrows():
                name = row['Name']
                if name and not row.iloc[1:].isna().all():  # Check if name exists and not all other fields are empty
                    config = row.to_dict()
                    config.pop('Name')
                    updated_config[name] = config
            
            # Update the config_manager with the new configuration
            config_manager.size_config = updated_config
            config_manager.save_size_config()
            
            # Create a new dataframe with the updated configuration
            new_df = pd.DataFrame(
                [[name] + list(config.values()) for name, config in updated_config.items()],
                columns=['Name'] + (list(next(iter(updated_config.values())).keys()) if updated_config else [])
            )
            
            return new_df, t('size_config_updated', config_manager.language)

        def add_color_config(df):
            """Add a new empty row to the color configuration table."""
            new_row = pd.DataFrame([['' for _ in df.columns]], columns=df.columns)
            updated_df = pd.concat([df, new_row], ignore_index=True)
            return updated_df, t('color_config_row_added', config_manager.language)

        def update_color_config(df):
            """Save all changes made to the color configuration table and remove empty rows."""
            updated_config = {}
            for _, row in df.iterrows():
                name = row['Name']
                if name and not row.iloc[1:].isna().all():  # Check if name exists and not all other fields are empty
                    config = row.to_dict()
                    config.pop('Name')
                    updated_config[name] = config
            
            # Update the config_manager with the new configuration
            config_manager.color_config = updated_config
            config_manager.save_color_config()
            
            # Create a new dataframe with the updated configuration
            new_df = pd.DataFrame(
                [[name] + list(config.values()) for name, config in updated_config.items()],
                columns=['Name', 'R', 'G', 'B', 'Notes']
            )
            
            return new_df, t('color_config_updated', config_manager.language)

        def update_size_input_visibility(use_csv_val, option_type_val, lang_val):
            _size_option_choices_translated = [
                (t('target_size_radio', lang_val), "target"), 
                (t('size_range_radio', lang_val), "range")
            ]
            current_target_size_label = t('target_size', lang_val)
            current_size_range_min_label = t('size_range_min', lang_val)
            current_size_range_max_label = t('size_range_max', lang_val)
            current_size_option_label = t('size_input_option', lang_val)

            updates = {
                size_option_type: gr.update(choices=_size_option_choices_translated, label=current_size_option_label)
            }

            if use_csv_val:
                # If CSV size is used, hide and clear all custom size options
                updates.update({
                    size_option_type: gr.update(visible=False, choices=_size_option_choices_translated, label=current_size_option_label),
                    target_size: gr.update(visible=False, value=None, label=current_target_size_label),
                    size_range_min: gr.update(visible=False, value=None, label=current_size_range_min_label),
                    size_range_max: gr.update(visible=False, value=None, label=current_size_range_max_label)
                })
            else:
                # If CSV size is NOT used, show radio and relevant inputs, and clear the hidden ones
                updates[size_option_type] = gr.update(visible=True, choices=_size_option_choices_translated, label=current_size_option_label)
                if option_type_val == "target":
                    updates.update({
                        target_size: gr.update(visible=True, label=current_target_size_label),
                        size_range_min: gr.update(visible=False, value=None, label=current_size_range_min_label),
                        size_range_max: gr.update(visible=False, value=None, label=current_size_range_max_label),
                    })
                else:  # "range"
                    updates.update({
                        target_size: gr.update(visible=False, value=None, label=current_target_size_label),
                        size_range_min: gr.update(visible=True, label=current_size_range_min_label),
                        size_range_max: gr.update(visible=True, label=current_size_range_max_label),
                    })
            return updates
        
        lang_outputs = [title, input_image, lang_dropdown, photo_type, photo_sheet_size, preset_color, background_color,
                sheet_rows, sheet_cols, photos_spacing, layout_only, yolov8_path, yunet_path, rmbg_path, size_config, color_config,
                compress, change_background, rotate, resize, add_crop_lines, use_csv_size, target_size, size_range_min, size_range_max,
                process_btn, output_image, corrected_output, download_final_file, download_corrected_file,
                notification, key_param_tab, advanced_settings_tab, config_management_tab, confirm_advanced_settings,save_final_btn, save_final_path,
                save_corrected_btn, save_corrected_path,
                size_config_tab, color_config_tab, result_tab, corrected_image_tab,
                size_df, color_df, add_size_btn, update_size_btn,
                add_color_btn, update_color_btn, config_notification,
                size_option_type, language, layout_position]

        lang_dropdown.change(
            update_language,
            inputs=[lang_dropdown],
            outputs=lang_outputs
        )

        confirm_advanced_settings.click(
            confirm_advanced_settings_fn,
            inputs=[yolov8_path, yunet_path, rmbg_path, size_config, color_config],
            outputs=[size_df, color_df, photo_type, photo_sheet_size, preset_color]
        )

        preset_color.change(
            update_background_color,
            inputs=[preset_color, lang_dropdown],
            outputs=[background_color]
        )

        background_color.change(
            update_preset_color,
            inputs=[background_color, lang_dropdown],
            outputs=[preset_color]
        )

        # Background color preview: re-composite cached transparent result (no model re-run)
        background_color.change(
            update_background_preview,
            inputs=[
                background_color, photo_type, photo_sheet_size, compress, use_csv_size,
                target_size, size_range_min, size_range_max, change_background, layout_only,
                rotate, sheet_rows, sheet_cols, add_crop_lines, layout_position, photos_spacing,
                lang_dropdown, batch_corrected_images_alpha, batch_processed_paths,
                batch_final_images, batch_corrected_images, batch_download_offset, batch_temp_dir
            ],
            outputs=[
                output_image, corrected_output, download_final_file, download_corrected_file,
                batch_final_images, batch_corrected_images, batch_corrected_images_alpha, batch_processed_paths,
                batch_download_offset, batch_temp_dir
            ]
        )

        add_size_btn.click(add_size_config, inputs=[size_df], outputs=[size_df, config_notification])
        update_size_btn.click(update_size_config, inputs=[size_df], outputs=[size_df, config_notification])

        add_color_btn.click(add_color_config, inputs=[color_df], outputs=[color_df, config_notification])
        update_color_btn.click(update_color_config, inputs=[color_df], outputs=[color_df, config_notification])

        use_csv_size.change(
            fn=update_size_input_visibility,
            inputs=[use_csv_size, size_option_type, language],
            outputs=[size_option_type, target_size, size_range_min, size_range_max]
        )

        size_option_type.change(
            fn=update_size_input_visibility,
            inputs=[use_csv_size, size_option_type, language],
            outputs=[size_option_type, target_size, size_range_min, size_range_max]
        )

        process_btn.click(
            process_and_display_wrapper,
            inputs=[input_image, yolov8_path, yunet_path, rmbg_path, size_config, color_config,
                    photo_type, photo_sheet_size, background_color, compress, change_background,
                    rotate, resize, sheet_rows, sheet_cols, layout_only, add_crop_lines,
                    target_size, size_range_min, size_range_max, use_csv_size, layout_position, photos_spacing,
                    lang_dropdown, batch_temp_dir],
            outputs=[output_image, corrected_output, download_final_file, download_corrected_file,
                     batch_final_images, batch_corrected_images, batch_corrected_images_alpha, batch_processed_paths,
                     batch_download_offset, batch_temp_dir]
        )

        def select_batch_result(evt: gr.SelectData, final_images, corrected_images, offset):
            if not final_images or not corrected_images:
                return None, None

            index = evt.index - int(offset or 0)
            if index < 0:
                index = 0
            index = min(index, len(final_images) - 1)

            corrected_index = min(index, len(corrected_images) - 1)
            return final_images[index], corrected_images[corrected_index]

        download_final_file.select(
            fn=select_batch_result,
            inputs=[batch_final_images, batch_corrected_images, batch_download_offset],
            outputs=[output_image, corrected_output]
        )

        download_corrected_file.select(
            fn=select_batch_result,
            inputs=[batch_final_images, batch_corrected_images, batch_download_offset],
            outputs=[output_image, corrected_output]
        )

        def save_image_handler_wrapper(image, path, lang, photo_type, photo_sheet_size, background_color, compress, use_csv_size, target_size, size_range_min, size_range_max, is_corrected):
            return save_image_handler(
                image, path, lang, photo_type, photo_sheet_size, background_color,
                compress, use_csv_size, target_size, size_range_min, size_range_max,
                is_corrected, is_download_mode=False
            )
        
        final_save_fn = partial(save_image_handler_wrapper, is_corrected=False)
        corrected_save_fn = partial(save_image_handler_wrapper, is_corrected=True)

        save_final_btn.click(
            final_save_fn,
            inputs=[output_image, save_final_path, lang_dropdown, photo_type, photo_sheet_size, background_color, compress, use_csv_size, target_size, size_range_min, size_range_max],
            outputs=[notification]
        )

        save_corrected_btn.click(
            corrected_save_fn,
            inputs=[corrected_output, save_corrected_path, lang_dropdown, photo_type, photo_sheet_size, background_color, compress, use_csv_size, target_size, size_range_min, size_range_max],
            outputs=[notification]
        )
    
    return demo

def get_config_value(env_var_name, default_value, value_type=str, choices=None):
    """
    Get configuration value from environment variable with priority: CLI > ENV > auto-detection > default
    
    :param env_var_name: Environment variable name (without LIYING_ prefix)
    :param default_value: Default value if not found in environment
    :param value_type: Type to convert the value to (str, int)
    :param choices: List of valid choices for validation
    :return: Configuration value
    """
    env_key = f"LIYING_{env_var_name.upper()}"
    env_value = os.environ.get(env_key)
    
    if env_value:
        try:
            if value_type == int:
                result = int(env_value)
            else:
                result = env_value
            
            if choices and result not in choices:
                print(f"Warning: {env_key}={env_value} is not in valid choices {choices}, using default value")
                return default_value
            
            return result
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to parse {env_key}={env_value}, using default value: {e}")
            return default_value
    
    return default_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiYing Photo Processing System")
    parser.add_argument("--lang", type=str, choices=['en', 'zh'], default=get_config_value('lang', get_language(), str, ['en', 'zh']),
                        help="Specify the language (en/zh)")
    parser.add_argument("--server_name", type=str, default=get_config_value('server_name', "127.0.0.1"), help="Specify the hostname or IP address the server should bind to (default: 127.0.0.1)")
    parser.add_argument("--server_port", type=int, default=get_config_value('server_port', 7860, int), help="Specify the port number the server should listen on (default: 7860)")
    parser.add_argument("--deployment_mode", type=str, choices=['local', 'server'], default=None,
                        help="Specify the deployment mode (local/server). If not specified, auto-detect based on server_name")
    args = parser.parse_args()

    if args.deployment_mode:
        deployment_mode = args.deployment_mode
    else:
        deployment_mode_env = os.environ.get('LIYING_DEPLOYMENT_MODE')
        if deployment_mode_env and deployment_mode_env in ['local', 'server']:
            deployment_mode = deployment_mode_env
        else:
            if deployment_mode_env:
                print(f"Warning: LIYING_DEPLOYMENT_MODE={deployment_mode_env} is not valid, auto-detecting...")
            deployment_mode = 'local' if args.server_name in ['127.0.0.1', 'localhost'] else 'server'

    initial_language = args.lang
    demo = create_demo(initial_language, deployment_mode)

    print(f"Starting Gradio server on {args.server_name}:{args.server_port}")
    print(f"Deployment mode: {deployment_mode}")
    demo.launch(share=False, server_name=args.server_name, server_port=args.server_port, ssl_verify=False)
