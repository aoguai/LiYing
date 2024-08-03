import configparser
import os


class PhotoRequirements:
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'data', 'data.ini'
            )

        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"配置文件 {config_file} 不存在")

        self.config_file = config_file
        self.config = configparser.ConfigParser()
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                self.config.read_file(file)
        except Exception as e:
            raise IOError(f"读取配置文件时出错: {e}")

    def get_requirements(self, photo_type):
        if not isinstance(photo_type, str):
            raise TypeError("photo_type必须是字符串。")

        if photo_type in self.config:
            requirements = self.config[photo_type]
            return {
                '打印尺寸': requirements.get('打印尺寸', 'N/A'),
                '电子版尺寸': requirements.get('电子版尺寸', 'N/A'),
                '分辨率': requirements.get('分辨率', 'N/A'),
                '文件格式': requirements.get('文件格式', 'N/A'),
                '文件大小': requirements.get('文件大小', 'N/A')
            }
        else:
            return None

    def list_photo_types(self):
        return self.config.sections()

    def get_resize_image_list(self, photo_type):
        requirements = self.get_requirements(photo_type)
        if not requirements:
            print("未找到指定的照片类型。")
            return None

        electronic_size = requirements['电子版尺寸'].replace("dpi", "")
        if electronic_size == 'N/A':
            return "300"

        try:
            width, height = map(int, electronic_size.replace("px", "").split(' x '))
        except ValueError:
            raise ValueError(f"电子尺寸格式无效: {electronic_size}")

        return [width, height, electronic_size]