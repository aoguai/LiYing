import configparser
import locale
import os


class PhotoRequirements:
    def __init__(self, config_file=None):
        # Get custom language environment variable
        language = os.getenv('CLI_LANGUAGE', '')
        if language == '':
            # Get system language
            system_language, _ = locale.getdefaultlocale()
            language = 'en' if system_language and system_language.startswith('en') else 'zh'

        if config_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if language == 'en':
                config_file = os.path.join(base_dir, 'data', 'data_en.ini')
            else:
                config_file = os.path.join(base_dir, 'data', 'data_zh.ini')

        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"The configuration file {config_file} does not exist")

        self.config_file = config_file
        self.config = configparser.ConfigParser()
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                self.config.read_file(file)
        except Exception as e:
            raise IOError(f"Error reading configuration file: {e}")

    def get_requirements(self, photo_type):
        if not isinstance(photo_type, str):
            raise TypeError("Photo_date must be a string.")
        if photo_type in self.config:
            requirements = self.config[photo_type]
            return {
                'print_size': requirements.get('print_size', requirements.get('打印尺寸', 'N/A')),
                'electronic_size': requirements.get('electronic_size', requirements.get('电子版尺寸', 'N/A')),
                'resolution': requirements.get('resolution', requirements.get('分辨率', 'N/A')),
                'file_format': requirements.get('file_format', requirements.get('文件格式', 'N/A')),
                'file_size': requirements.get('file_size', requirements.get('文件大小', 'N/A'))
            }
        else:
            return None

    def list_photo_types(self):
        return self.config.sections()

    def get_resize_image_list(self, photo_type):
        requirements = self.get_requirements(photo_type)
        if not requirements:
            try:
                electronic_size = photo_type.replace("dpi", "")
            except ValueError:
                raise ValueError(f"Invalid electronic_size format: {photo_type}")
        else:
            electronic_size = requirements['electronic_size'].replace("dpi", "")
            if electronic_size == 'N/A':
                return [None, None, None]

        try:
            width, height = map(int, electronic_size.replace("px", "").split(' x '))
        except ValueError:
            raise ValueError(f"Invalid electronic_size format: {electronic_size}")

        return [width, height, electronic_size]
