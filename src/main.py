import locale

import click
import os
import sys

# Set the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
TOOL_DIR = os.path.join(PROJECT_ROOT, 'tool')
# Add data and model directories to sys.path
sys.path.append(DATA_DIR)
sys.path.append(MODEL_DIR)
sys.path.append(TOOL_DIR)

# Set src directory as PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from tool.ImageProcessor import ImageProcessor
from tool.PhotoSheetGenerator import PhotoSheetGenerator


class BGRListType(click.ParamType):
    name = 'bgr_list'

    def convert(self, value, param, ctx):
        if value:
            try:
                return tuple(float(x) for x in value.split(','))
            except ValueError:
                self.fail(f'{value} is not a valid BGR list format. Expected format: FLOAT,FLOAT,FLOAT.')
        return 1.0, 1.0, 1.0  # Default value


def get_language():
    # Get custom language environment variable
    language = os.getenv('CLI_LANGUAGE', '')
    if language == '':
        # Get system language
        system_language, _ = locale.getdefaultlocale()
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
              default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/yolov8n-pose.onnx'),
              help='Path to YOLOv8 model' if get_language() == 'en' else 'YOLOv8 模型路径')
@click.option('-u', '--yunet-model-path', type=click.Path(),
              default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'model/face_detection_yunet_2023mar.onnx'),
              help='Path to YuNet model' if get_language() == 'en' else 'YuNet 模型路径')
@click.option('-r', '--rmbg-model-path', type=click.Path(),
              default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/RMBG-1.4-model.onnx'),
              help='Path to RMBG model' if get_language() == 'en' else 'RMBG 模型路径')
@click.option('-b', '--bgr-list', type=BGRListType(), default='1.0,1.0,1.0',
              help='BGR channel values list (comma-separated) for image composition' if get_language() == 'en' else 'BGR 通道值列表（逗号分隔），用于图像合成')
@click.option('-s', '--save-path', type=click.Path(), default='output.jpg',
              help='Path to save the output image' if get_language() == 'en' else '保存路径')
@click.option('-p', '--photo-type', type=str, default='一寸照片',
              help='Type of photo' if get_language() == 'en' else '照片类型')
@click.option('--photo-sheet-size', type=click.Choice(['5', '6'], case_sensitive=False), default='5',
              help='Size of the photo sheet (5-inch or 6-inch)' if get_language() == 'en' else '选择照片表格的尺寸（五寸或六寸）')
@click.option('-c', '--compress/--no-compress', default=False,
              help='Whether to compress the image' if get_language() == 'en' else '是否压缩图像')
@click.option('-sc', '--save-corrected/--no-save-corrected', default=False,
              help='Whether to save the corrected image' if get_language() == 'en' else '是否保存修正图像后的图片')
@click.option('-bg', '--change-background/--no-change-background', default=False,
              help='Whether to change the background' if get_language() == 'en' else '是否替换背景')
@click.option('-sb', '--save-background/--no-save-background', default=False,
              help='Whether to save the image with changed background' if get_language() == 'en' else '是否保存替换背景后的图像')
@click.option('-sr', '--sheet-rows', type=int, default=3,
              help='Number of rows in the photo sheet' if get_language() == 'en' else '照片表格的行数')
@click.option('-sc', '--sheet-cols', type=int, default=3,
              help='Number of columns in the photo sheet' if get_language() == 'en' else '照片表格的列数')
@click.option('--rotate/--no-rotate', default=False,
              help='Whether to rotate the photo by 90 degrees' if get_language() == 'en' else '是否旋转照片90度')
@click.option('-rs', '--resize/--no-resize', default=True,
              help='Whether to resize the image' if get_language() == 'en' else '是否调整图像尺寸')
@click.option('-srz', '--save-resized/--no-save-resized', default=False,
              help='Whether to save the resized image' if get_language() == 'en' else '是否保存调整尺寸后的图像')
def cli(img_path, yolov8_model_path, yunet_model_path, rmbg_model_path, bgr_list, save_path, photo_type,
        photo_sheet_size, compress, save_corrected,
        change_background, save_background, sheet_rows, sheet_cols, rotate, resize, save_resized):
    # Create an instance of the image processor
    processor = ImageProcessor(img_path, yolov8_model_path, yunet_model_path, rmbg_model_path, bgr_list, y_b=compress)
    # Crop and correct image
    processor.crop_and_correct_image()
    if save_corrected:
        corrected_path = os.path.splitext(save_path)[0] + '_corrected' + os.path.splitext(save_path)[1]
        processor.save_photos(corrected_path, compress)
        echo_message('corrected_saved', path=corrected_path)

    # Optional background change
    if change_background:
        processor.change_background()
        if save_background:
            background_path = os.path.splitext(save_path)[0] + '_background' + os.path.splitext(save_path)[1]
            processor.save_photos(background_path, compress)
            echo_message('background_saved', path=background_path)

    # Optional resizing
    if resize:
        processor.resize_image(photo_type)
        if save_resized:
            resized_path = os.path.splitext(save_path)[0] + '_resized' + os.path.splitext(save_path)[1]
            processor.save_photos(resized_path, compress)
            echo_message('resized_saved', path=resized_path)

    # Generate photo sheet
    # Set photo sheet size
    if photo_sheet_size == '5':
        sheet_width, sheet_height = 1050, 1500
    else:
        sheet_width, sheet_height = 1300, 1950
    generator = PhotoSheetGenerator([sheet_width, sheet_height])
    photo_sheet_cv = generator.generate_photo_sheet(processor.photo.image, sheet_rows, sheet_cols, rotate)
    sheet_path = os.path.splitext(save_path)[0] + '_sheet' + os.path.splitext(save_path)[1]
    generator.save_photo_sheet(photo_sheet_cv, sheet_path)
    echo_message('sheet_saved', path=sheet_path)


if __name__ == "__main__":
    cli()
