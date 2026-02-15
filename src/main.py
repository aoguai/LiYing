import locale

import click
import os
import sys
import warnings

def get_app_path():
    """Get the application base path for both dev and PyInstaller environments."""
    if getattr(sys, 'frozen', False):
        # In PyInstaller bundle
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_root_path():
    """Get the root path for both dev and PyInstaller environments."""
    if getattr(sys, 'frozen', False):
        # In PyInstaller bundle
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set the project paths
APP_PATH = get_app_path()
ROOT_PATH = get_root_path()
DATA_DIR = os.path.join(ROOT_PATH, 'data')
MODEL_DIR = os.path.join(APP_PATH, 'model')
TOOL_DIR = os.path.join(APP_PATH, 'tool')

# Add data and model directories to sys.path
sys.path.extend([DATA_DIR, MODEL_DIR, TOOL_DIR])

# Set src directory as PYTHONPATH
sys.path.insert(0, APP_PATH)

from tool.ImageProcessor import ImageProcessor
from tool.PhotoSheetGenerator import PhotoSheetGenerator

from tool.PhotoRequirements import PhotoRequirements


class RGBListType(click.ParamType):
    name = 'rgb_list'

    def convert(self, value, param, ctx):
        if value:
            try:
                return tuple(int(x) for x in value.split(','))
            except ValueError:
                self.fail(f'{value} is not a valid RGB/RGBA list format. Expected format: INTEGER,INTEGER,INTEGER[,INTEGER].')
        return 0, 0, 0  # Default value


class SizeRangeType(click.ParamType):
    name = 'size_range'

    def convert(self, value, param, ctx):
        if value:
            try:
                min_size, max_size = map(int, value.split(','))
                if min_size <= 0 or max_size <= 0:
                    self.fail(f'Size range values must be greater than 0, got min_size={min_size}, max_size={max_size}')
                if min_size >= max_size:
                    self.fail(f'Minimum size must be less than maximum size, got min_size={min_size}, max_size={max_size}')
                return min_size, max_size
            except ValueError:
                self.fail(f'{value} is not a valid size range format. Expected format: MIN_SIZE,MAX_SIZE.')
        return None


def get_language():
    # Get custom language environment variable
    language = os.getenv('CLI_LANGUAGE', '')
    if language == '':
        # Get system language
        try:
            # Try to get current locale
            current_locale = locale.getlocale()[0]
            if current_locale is None:
                # If no locale is set, try to set the default locale
                locale.setlocale(locale.LC_ALL, '')
                current_locale = locale.getlocale()[0]
            
            # Extract language code from locale
            if current_locale:
                system_language = current_locale.split('_')[0]
            else:
                system_language = 'en'
        except Exception:
            system_language = 'en'
            
        language = 'en' if system_language and system_language.startswith('en') else 'zh'
        return language
    return language


# Define multilingual support messages
messages = {
    'en': {
        'corrected_saved': 'Corrected image saved to {path}',
        'background_saved': 'Background-changed image saved to {path}',
        'resized_saved': 'Resized image saved to {path}',
        'sheet_saved': 'Photo sheet saved to {path}',
    },
    'zh': {
        'corrected_saved': '裁剪并修正后的图像已保存到 {path}',
        'background_saved': '替换背景后的图像已保存到 {path}',
        'resized_saved': '调整尺寸后的图像已保存到 {path}',
        'sheet_saved': '照片表格已保存到 {path}',
    }
}


def echo_message(key, **kwargs):
    lang = get_language()
    message = messages.get(lang, messages['en']).get(key, '')
    click.echo(message.format(**kwargs))


@click.command()
@click.argument('img_path', type=click.Path(exists=True, resolve_path=True))
@click.option('-y', '--yolov8-model-path', type=click.Path(),
              default=os.path.join(MODEL_DIR, 'yolov8n-pose.onnx'),
              help='Path to YOLOv8 model' if get_language() == 'en' else 'YOLOv8 模型路径')
@click.option('-u', '--yunet-model-path', type=click.Path(),
              default=os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx'),
              help='Path to YuNet model' if get_language() == 'en' else 'YuNet 模型路径')
@click.option('-r', '--rmbg-model-path', type=click.Path(),
              default=os.path.join(MODEL_DIR, 'RMBG-1.4-model.onnx'),
              help='Path to RMBG model' if get_language() == 'en' else 'RMBG 模型路径')
@click.option('-sz', '--size-config', type=click.Path(exists=True),
              default=os.path.join(DATA_DIR, f'size_{get_language()}.csv'),
              help='Path to size configuration file' if get_language() == 'en' else '尺寸配置文件路径')
@click.option('-cl', '--color-config', type=click.Path(exists=True),
              default=os.path.join(DATA_DIR, f'color_{get_language()}.csv'),
              help='Path to color configuration file' if get_language() == 'en' else '颜色配置文件路径')
@click.option('-b', '--rgb-list', type=RGBListType(), default='0,0,0',
              help='RGB(A) channel values list (comma-separated) for image composition (optional alpha: 0-255)' if get_language() == 'en' else 'RGB(A) 通道值列表（英文逗号分隔，可选 Alpha: 0-255），用于图像合成')
@click.option('-s', '--save-path', type=click.Path(), default='output.jpg',
              help='Path to save the output image' if get_language() == 'en' else '保存路径')
@click.option('-p', '--photo-type', type=str, default='One Inch' if get_language() == 'en' else '一寸',
              help='Photo types' if get_language() == 'en' else '照片类型')
@click.option('-ps', '--photo-sheet-size', type=str, default='Five Inch' if get_language() == 'en' else '五寸',
              help='Size of the photo sheet' if get_language() == 'en' else '选择照片表格的尺寸')
@click.option('-c', '--compress/--no-compress', default=False,
              help='Whether to compress the image' if get_language() == 'en' else '是否压缩图像（使用 AGPicCompress 压缩）')
@click.option('-sv', '--save-corrected/--no-save-corrected', default=False,
              help='Whether to save the corrected image' if get_language() == 'en' else '是否保存修正图像后的图片')
@click.option('-bg', '--change-background/--no-change-background', default=False,
              help='Whether to change the background' if get_language() == 'en' else '是否替换背景')
@click.option('-sb', '--save-background/--no-save-background', default=False,
              help='Whether to save the image with changed background' if get_language() == 'en' else '是否保存替换背景后的图像')
@click.option('-lo', '--layout-only', is_flag=True, default=False,
              help='Only layout the photo without changing background' if get_language() == 'en' else '仅排版照片，不更换背景')
@click.option('-sr', '--sheet-rows', type=int, default=3,
              help='Number of rows in the photo sheet' if get_language() == 'en' else '照片表格的行数')
@click.option('-sc', '--sheet-cols', type=int, default=3,
              help='Number of columns in the photo sheet' if get_language() == 'en' else '照片表格的列数')
@click.option('-rt', '--rotate/--no-rotate', default=False,
              help='Whether to rotate the photo by 90 degrees' if get_language() == 'en' else '是否旋转照片90度')
@click.option('-rs', '--resize/--no-resize', default=True,
              help='Whether to resize the image' if get_language() == 'en' else '是否调整图像尺寸')
@click.option('-svr', '--save-resized/--no-save-resized', default=False,
              help='Whether to save the resized image' if get_language() == 'en' else '是否保存调整尺寸后的图像')
@click.option('-al', '--add-crop-lines/--no-add-crop-lines', default=True, 
              help='Add crop lines to the photo sheet' if get_language() == 'en' else '在照片表格上添加裁剪线')
@click.option('-ts', '--target-size', type=int,
              help='Target file size in KB. When specified, ignores quality and size-range.' if get_language() == 'en' else '目标文件大小（KB）。指定后将忽略质量和大小范围参数。')
@click.option('-szr', '--size-range', type=SizeRangeType(),
              help='File size range in KB as min,max (e.g., 10,20)' if get_language() == 'en' else '文件大小范围（KB），格式为最小值,最大值（例如：10,20）')
@click.option('-uc', '--use-csv-size/--no-use-csv-size', default=True,
              help='Whether to use file size limits from CSV' if get_language() == 'en' else '是否使用CSV中的文件大小限制')
@click.option('-lp', '--layout-position', type=click.IntRange(0, 8), default=4,
              help='Layout position (0-8): 0=top-left, 1=top, 2=top-right, 3=middle-left, 4=center, 5=middle-right, 6=bottom-left, 7=bottom, 8=bottom-right' if get_language() == 'en' else '布局位置(0-8)：0=左上，1=上，2=右上，3=左中，4=中，5=右中，6=左下，7=下，8=右下')
@click.option('-psp', '--photos-spacing', type=int, default=0, help='Pixel spacing between photos in the sheet (default: 0)' if get_language() == 'en' else '照片间距（像素，默认0）')
def cli(img_path, yolov8_model_path, yunet_model_path, rmbg_model_path, size_config, color_config, rgb_list, save_path, 
        photo_type, photo_sheet_size, compress, save_corrected, change_background, save_background, layout_only, sheet_rows, 
        sheet_cols, rotate, resize, save_resized, add_crop_lines, target_size, size_range, use_csv_size, layout_position, photos_spacing):
    # Parameter validation
    if target_size is not None and size_range is not None:
        warnings.warn("Both target_size and size_range are provided. Using target_size and ignoring size_range.")
        size_range = None
        
    # Create an instance of the image processor
    processor = ImageProcessor(img_path, yolov8_model_path, yunet_model_path, rmbg_model_path, rgb_list, y_b=compress)
    photo_requirements = PhotoRequirements(get_language(), size_config, color_config)
    
    # Get file size limits from CSV if enabled
    file_size_limits = {}
    if use_csv_size:
        file_size_limits = photo_requirements.get_file_size_limits(photo_type)
        
    # User-provided limits override CSV limits
    if target_size is not None:
        file_size_limits = {'target_size': target_size}
    elif size_range is not None:
        file_size_limits = {'size_range': size_range}
    
    # Crop and correct image
    processor.crop_and_correct_image()
    if save_corrected:
        corrected_path = os.path.splitext(save_path)[0] + '_corrected' + os.path.splitext(save_path)[1]
        processor.save_photos(corrected_path, compress, **file_size_limits)
        echo_message('corrected_saved', path=corrected_path)

    # Optional background change
    if change_background and not layout_only:
        processor.change_background()
        if save_background:
            background_path = os.path.splitext(save_path)[0] + '_background' + os.path.splitext(save_path)[1]
            processor.save_photos(background_path, compress, **file_size_limits)
            echo_message('background_saved', path=background_path)

    # Optional resizing
    if resize:
        processor.resize_image(photo_type)
        if save_resized:
            resized_path = os.path.splitext(save_path)[0] + '_resized' + os.path.splitext(save_path)[1]
            processor.save_photos(resized_path, compress, **file_size_limits)
            echo_message('resized_saved', path=resized_path)

    # Generate photo sheet
    # Set photo sheet size
    sheet_info = photo_requirements.get_resize_image_list(photo_sheet_size)
    sheet_width, sheet_height, sheet_resolution = sheet_info['width'], sheet_info['height'], sheet_info['resolution']
    generator = PhotoSheetGenerator((sheet_width, sheet_height), sheet_resolution)
    photo_sheet_cv = generator.generate_photo_sheet(processor.photo.image, sheet_rows, sheet_cols, rotate, add_crop_lines, layout_position, photos_spacing)
    sheet_path = os.path.splitext(save_path)[0] + '_sheet' + os.path.splitext(save_path)[1]
    generator.save_photo_sheet(photo_sheet_cv, sheet_path)
    echo_message('sheet_saved', path=sheet_path)


if __name__ == "__main__":
    cli()
