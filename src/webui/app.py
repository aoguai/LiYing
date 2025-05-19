import argparse
import json
import locale
import os
import re
import sys
import time

import cv2
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

sys.path.extend([DATA_DIR, MODEL_DIR, TOOL_DIR])

from tool.ImageProcessor import ImageProcessor
from tool.PhotoSheetGenerator import PhotoSheetGenerator
from tool.PhotoRequirements import PhotoRequirements
from tool.ConfigManager import ConfigManager

def get_language():
    """Get the system language or default to English."""
    try:
        system_lang = locale.getlocale()[0].split('_')[0]
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

def parse_color(color_string):
    """Parse color string to RGB list."""
    if color_string is None:
        return [255, 255, 255]
    if color_string.startswith('#'):
        return [int(color_string.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
    rgb_match = re.match(r'rgba?$(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)(?:,\s*[\d.]+)?$', color_string)
    if rgb_match:
        return [min(255, max(0, int(float(x)))) for x in rgb_match.groups()]
    return [255, 255, 255]

def process_image(img_path, yolov8_path, yunet_path, rmbg_path, photo_requirements, photo_type, photo_sheet_size, rgb_list, compress=False, change_background=False, rotate=False, resize=True, sheet_rows=3, sheet_cols=3, add_crop_lines=True, target_size=None, size_range=None, use_csv_size=True):
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
    if use_csv_size:
        file_size_limits = photo_requirements.get_file_size_limits(photo_type)
    
    # User-provided limits override CSV limits
    if target_size is not None:
        file_size_limits = {'target_size': target_size}
    elif size_range is not None and len(size_range) == 2:
        # Convert from list to tuple for size_range parameter
        file_size_limits = {'size_range': tuple(size_range)}

    if change_background:
        processor.change_background()

    if resize:
        processor.resize_image(photo_type)

    sheet_info = photo_requirements.get_resize_image_list(photo_sheet_size)
    sheet_width, sheet_height, sheet_resolution = sheet_info['width'], sheet_info['height'], sheet_info['resolution']
    generator = PhotoSheetGenerator((sheet_width, sheet_height), sheet_resolution)
    photo_sheet_cv = generator.generate_photo_sheet(processor.photo.image, sheet_rows, sheet_cols, rotate, add_crop_lines)

    return {
        'final_image': photo_sheet_cv,
        'corrected_image': processor.photo.image,
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


def create_demo(initial_language):
    """Create the Gradio demo interface."""
    config_manager = ConfigManager(language=initial_language)
    config_manager.load_configs()
    photo_requirements = PhotoRequirements(language=initial_language)

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

        title = gr.Markdown(f"# {t('title', initial_language)}")

        color_change_source = {"source": "custom"}

        current_file_format = 'png'
        current_resolution = 300

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label=t('upload_photo', initial_language), height=400)
                lang_dropdown = gr.Dropdown(
                    choices=[("English", "en"), ("中文", "zh")], 
                    value=initial_language, 
                    label=t('language', initial_language)
                )

                with gr.Tabs() as tabs:
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
                        with gr.Row():
                            preset_color = gr.Dropdown(choices=color_choices, label=t('preset_color', initial_language), value=t('custom_color', initial_language))
                            background_color = gr.ColorPicker(label=t('background_color', initial_language), value="#FFFFFF")
                        layout_only = gr.Checkbox(label=t('layout_only', initial_language), value=False)
                        sheet_rows = gr.Slider(minimum=1, maximum=10, step=1, value=3, label=t('sheet_rows', initial_language))
                        sheet_cols = gr.Slider(minimum=1, maximum=10, step=1, value=3, label=t('sheet_cols', initial_language))
                    
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
                        with gr.Tabs() as config_tabs:
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
                with gr.Tabs() as result_tabs:
                    with gr.TabItem(t('result', initial_language)) as result_tab:
                        output_image = gr.Image(label=t('final_image', initial_language), height=800)
                        with gr.Row():
                            save_final_btn = gr.Button(t('save_image', initial_language))
                            save_final_path = gr.Textbox(label=t('save_path', initial_language), value=SAVE_IMG_DIR)
                    with gr.TabItem(t('corrected_image', initial_language)) as corrected_image_tab:
                        corrected_output = gr.Image(label=t('corrected_image', initial_language), height=800)
                        with gr.Row():
                            save_corrected_btn = gr.Button(t('save_corrected', initial_language))
                            save_corrected_path = gr.Textbox(label=t('save_path', initial_language), value=SAVE_IMG_DIR)
                notification = gr.Textbox(label=t('notification', initial_language))

        def process_and_display_wrapper(input_image, yolov8_path, yunet_path, rmbg_path, size_config, color_config, 
                                        photo_type, photo_sheet_size, background_color, compress, change_background, 
                                        rotate, resize, sheet_rows, sheet_cols, layout_only, add_crop_lines, 
                                        target_size, size_range_min, size_range_max, use_csv_size):
            nonlocal current_file_format, current_resolution

            validated_target_size = None
            validated_size_range = None

            if target_size is not None and target_size > 0:
                validated_target_size = int(target_size)

            if size_range_min is not None and size_range_max is not None and size_range_min > 0 and size_range_max > 0:
                if size_range_min < size_range_max:
                    validated_size_range = [int(size_range_min), int(size_range_max)]
                else:
                    gr.Warning(t('size_range_error', language) if 'size_range_error' in TEXTS[language] else "Min size must be less than max size")

            if validated_target_size is not None and validated_size_range is not None:
                gr.Warning(t('size_params_conflict', language) if 'size_params_conflict' in TEXTS[language] else "Both target size and size range provided. Using target size.")
                validated_size_range = None
            
            result = process_and_display(
                input_image, yolov8_path, yunet_path, rmbg_path, size_config, color_config, 
                photo_type, photo_sheet_size, background_color, compress, change_background, 
                rotate, resize, sheet_rows, sheet_cols, layout_only, add_crop_lines,
                validated_target_size, validated_size_range, use_csv_size
            )
            
            if result:
                final_image, corrected_image, file_format, resolution = result
                current_file_format = file_format if file_format else 'png'
                current_resolution = resolution if resolution else 300
                return final_image, corrected_image
            return None, None

        def save_image_handler(image, path, lang, photo_type, photo_sheet_size, background_color):
            nonlocal current_file_format, current_resolution
            if image is None:
                return t('no_image_to_save', lang)

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
                base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)  # 仅过滤文件名非法字符
                filename = f"{base_name}.{current_file_format}"

                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                full_path = os.path.join(dir_name, filename) if dir_name else filename

            try:
                save_image(image, full_path, current_file_format, current_resolution)
                return t('image_saved_success', lang).format(path=full_path)
            except Exception as e:
                return t('image_save_error', lang).format(error=str(e))

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

            # The dictionary of updates to be returned
            updates = {
                title: gr.update(value=f"# {t('title', lang)}"),
                input_image: gr.update(label=t('upload_photo', lang)),
                lang_dropdown: gr.update(label=t('language', lang)),
                photo_type: gr.update(choices=new_photo_size_choices, label=t('photo_type', lang), value=new_photo_size_choices[0] if new_photo_size_choices else None),
                photo_sheet_size: gr.update(choices=new_sheet_size_choices, label=t('photo_sheet_size', lang), value=new_sheet_size_choices[0] if new_sheet_size_choices else None),
                preset_color: gr.update(choices=new_color_choices, label=t('preset_color', lang)),
                background_color: gr.update(label=t('background_color', lang)),
                layout_only: gr.update(label=t('layout_only', lang)),
                sheet_rows: gr.update(label=t('sheet_rows', lang)),
                sheet_cols: gr.update(label=t('sheet_cols', lang)),
                yolov8_path: gr.update(label=t('yolov8_path', lang)),
                yunet_path: gr.update(label=t('yunet_path', lang)),
                rmbg_path: gr.update(label=t('rmbg_path', lang)),
                size_config: gr.update(label=t('size_config', lang), value=DEFAULT_SIZE_CONFIG.format(lang)),
                color_config: gr.update(label=t('color_config', lang), value=DEFAULT_COLOR_CONFIG.format(lang)),
                compress: gr.update(label=t('compress', lang)),
                change_background: gr.update(label=t('change_background', lang)),
                rotate: gr.update(label=t('rotate', lang)),
                resize: gr.update(label=t('resize', lang)),
                add_crop_lines: gr.update(label=t('add_crop_lines', lang)),
                use_csv_size: gr.update(label=t('use_csv_size', lang)),
                target_size: gr.update(label=t('target_size', lang)),
                size_range_min: gr.update(label=t('size_range_min', lang)),
                size_range_max: gr.update(label=t('size_range_max', lang)),
                process_btn: gr.update(value=t('process_btn', lang)),
                output_image: gr.update(label=t('final_image', lang)),
                corrected_output: gr.update(label=t('corrected_image', lang)),
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
                corrected_image_tab: gr.update(label=t('corrected_image', lang)),
                size_df: gr.update(
                    value=pd.DataFrame(
                        [[name] + list(config.values()) for name, config in config_manager.size_config.items()],
                        columns=['Name'] + (list(next(iter(config_manager.size_config.values())).keys()) if config_manager.size_config else [])
                    ),
                    label=t('size_config_table', lang)
                ),
                color_df: gr.update(
                    value=pd.DataFrame(
                        [[name] + list(config.values()) for name, config in config_manager.color_config.items()],
                        columns=['Name', 'R', 'G', 'B', 'Notes']
                    ),
                    label=t('color_config_table', lang)
                ),
                add_size_btn: gr.update(value=t('add_size', lang)),
                update_size_btn: gr.update(value=t('save_size', lang)),
                add_color_btn: gr.update(value=t('add_color', lang)),
                update_color_btn: gr.update(value=t('save_color', lang)),
                config_notification: gr.update(label=t('config_notification', lang)),
                size_option_type: gr.update(
                    label=t('size_input_option', lang),
                    choices=[(t('target_size_radio', lang), "target"), (t('size_range_radio', lang), "range")]
                ),
            }
            # Add the language state update
            updates[language] = lang
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
            custom_color = t('custom_color', lang)
            return gr.update(value=custom_color)

        def process_and_display(image, yolov8_path, yunet_path, rmbg_path, size_config, color_config, photo_type, 
                                photo_sheet_size, background_color, compress, change_background, rotate, resize, 
                                sheet_rows, sheet_cols, layout_only, add_crop_lines, target_size=None, 
                                size_range=None, use_csv_size=True):
            """Process and display the image with given parameters."""
            rgb_list = parse_color(background_color)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            temp_image_path = "temp_input_image.jpg"
            cv2.imwrite(temp_image_path, image_bgr)

            result = process_image(
                temp_image_path,
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
                target_size=target_size,
                size_range=size_range,
                use_csv_size=use_csv_size
            )

            os.remove(temp_image_path)
            
            sheet_info = photo_requirements.get_resize_image_list(photo_sheet_size)
            file_format = sheet_info.get('file_format', 'png').lower()
            if file_format == 'jpg':
                file_format = 'jpeg'
            resolution = sheet_info.get('resolution', 300)
            
            final_image_rgb = cv2.cvtColor(result['final_image'], cv2.COLOR_BGR2RGB)
            corrected_image_rgb = cv2.cvtColor(result['corrected_image'], cv2.COLOR_BGR2RGB)

            return final_image_rgb, corrected_image_rgb, file_format, resolution


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

            if use_csv_val:
                # If CSV size is used, hide all custom size options
                return {
                    size_option_type: gr.update(visible=False, choices=_size_option_choices_translated, label=current_size_option_label),
                    target_size: gr.update(visible=False, label=current_target_size_label),
                    size_range_min: gr.update(visible=False, label=current_size_range_min_label),
                    size_range_max: gr.update(visible=False, label=current_size_range_max_label)
                }
            else:
                # If CSV size is NOT used, show radio and relevant inputs
                is_target_selected = option_type_val == "target"
                return {
                    size_option_type: gr.update(visible=True, choices=_size_option_choices_translated, label=current_size_option_label),
                    target_size: gr.update(visible=is_target_selected, label=current_target_size_label),
                    size_range_min: gr.update(visible=not is_target_selected, label=current_size_range_min_label),
                    size_range_max: gr.update(visible=not is_target_selected, label=current_size_range_max_label)
                }

        lang_dropdown.change(
            update_language,
            inputs=[lang_dropdown],
            outputs=[title, input_image, lang_dropdown, photo_type, photo_sheet_size, preset_color, background_color, 
                    sheet_rows, sheet_cols, layout_only, yolov8_path, yunet_path, rmbg_path, size_config, color_config, 
                    compress, change_background, rotate, resize, add_crop_lines, use_csv_size, target_size, size_range_min, size_range_max,
                    process_btn, output_image, 
                    corrected_output, notification, key_param_tab, advanced_settings_tab, config_management_tab, confirm_advanced_settings,save_final_btn, save_final_path,
                    save_corrected_btn, save_corrected_path,
                    size_config_tab, color_config_tab, result_tab, corrected_image_tab,
                    size_df, color_df, add_size_btn, update_size_btn,
                    add_color_btn, update_color_btn, config_notification,
                    size_option_type, language
                    ]
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
                    target_size, size_range_min, size_range_max, use_csv_size],
            outputs=[output_image, corrected_output]
        )
        
        save_final_btn.click(
            save_image_handler,
            inputs=[output_image, save_final_path, lang_dropdown, photo_type, photo_sheet_size, background_color],
            outputs=[notification]
        )

        save_corrected_btn.click(
            save_image_handler,
            inputs=[corrected_output, save_corrected_path, lang_dropdown, photo_type, photo_sheet_size, background_color],
            outputs=[notification]
        )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiYing Photo Processing System")
    parser.add_argument("--lang", type=str, choices=['en', 'zh'], default=get_language(), help="Specify the language (en/zh)")
    args = parser.parse_args()

    initial_language = args.lang
    demo = create_demo(initial_language)
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
