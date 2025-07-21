import json
import os
import sys
import unittest
import warnings

from PIL import Image

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'tool'))

from src.tool.ImageProcessor import ImageProcessor
from src.tool.PhotoSheetGenerator import PhotoSheetGenerator
from src.tool.PhotoRequirements import PhotoRequirements
from src.tool.ConfigManager import ConfigManager

class TestLiYing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_image_path = os.path.join(project_root, 'images', 'test1.jpg')
        cls.output_dir = os.path.join(project_root, 'tests', 'output')
        os.makedirs(cls.output_dir, exist_ok=True)

        # Set model paths
        cls.yolov8_model_path = os.path.join(project_root, 'src', 'model', 'yolov8n-pose.onnx')
        cls.yunet_model_path = os.path.join(project_root, 'src', 'model', 'face_detection_yunet_2023mar.onnx')
        cls.rmbg_model_path = os.path.join(project_root, 'src', 'model', 'RMBG-1.4-model.onnx')

    def get_first_valid_photo_size(self, photo_requirements):
        photo_types = photo_requirements.list_photo_types()
        return photo_types[0] if photo_types else None

    def check_models_exist(self):
        missing_models = []
        if not os.path.exists(self.yolov8_model_path):
            missing_models.append("YOLOv8")
        if not os.path.exists(self.yunet_model_path):
            missing_models.append("YuNet")
        if not os.path.exists(self.rmbg_model_path):
            missing_models.append("RMBG")
        return missing_models

    def check_image_dpi(self, image_path):
        with Image.open(image_path) as img:
            return img.info.get('dpi')

    def test_image_processor(self):
        missing_models = self.check_models_exist()
        if missing_models:
            warnings.warn(f"Skipping image processing test: Missing model files {', '.join(missing_models)}")
            return

        photo_requirements = PhotoRequirements(language='en')
        
        processor = ImageProcessor(self.test_image_path, 
                                yolov8_model_path=self.yolov8_model_path,
                                yunet_model_path=self.yunet_model_path,
                                RMBG_model_path=self.rmbg_model_path)
        processor.photo_requirements_detector = photo_requirements
        
        # Test crop and correct
        processor.crop_and_correct_image()
        self.assertIsNotNone(processor.photo.image)
        
        # Test background change
        processor.change_background([255, 0, 0])  # Red background, using list
        self.assertIsNotNone(processor.photo.image)
        
        # Test resize
        first_photo_size = self.get_first_valid_photo_size(photo_requirements)
        self.assertIsNotNone(first_photo_size, "No valid photo size found")
        processor.resize_image(first_photo_size)
        self.assertIsNotNone(processor.photo.image)
        
        # Save processed image
        output_path = os.path.join(self.output_dir, 'processed_image.jpg')
        processor.save_photos(output_path)
        self.assertTrue(os.path.exists(output_path))

        # Check DPI of saved image
        dpi = self.check_image_dpi(output_path)
        self.assertIsNotNone(dpi, "DPI information should be present in the saved image")
        self.assertEqual(dpi[0], dpi[1], "Horizontal and vertical DPI should be the same")

    def test_photo_sheet_generator(self):
        missing_models = self.check_models_exist()
        if missing_models:
            warnings.warn(f"Skipping photo sheet generation test: Missing model files {', '.join(missing_models)}")
            return

        photo_requirements = PhotoRequirements(language='en')
        
        processor = ImageProcessor(self.test_image_path,
                                yolov8_model_path=self.yolov8_model_path,
                                yunet_model_path=self.yunet_model_path,
                                RMBG_model_path=self.rmbg_model_path)
        processor.photo_requirements_detector = photo_requirements
        processor.crop_and_correct_image()
        
        first_photo_size = self.get_first_valid_photo_size(photo_requirements)
        self.assertIsNotNone(first_photo_size, "No valid photo size found")
        processor.resize_image(first_photo_size)
        
        # Use size from configuration
        sheet_sizes = photo_requirements.config_manager.get_sheet_sizes()
        first_sheet_size = next(iter(sheet_sizes))
        sheet_config = photo_requirements.config_manager.get_size_config(first_sheet_size)
        dpi = sheet_config.get('Resolution', 300)
        generator = PhotoSheetGenerator((sheet_config['ElectronicWidth'], sheet_config['ElectronicHeight']), dpi=dpi)
        sheet = generator.generate_photo_sheet(processor.photo.image, 2, 2)
        
        self.assertIsNotNone(sheet)
        self.assertEqual(sheet.shape[:2], (sheet_config['ElectronicHeight'], sheet_config['ElectronicWidth']))
        
        output_path = os.path.join(self.output_dir, 'photo_sheet.jpg')
        generator.save_photo_sheet(sheet, output_path)
        self.assertTrue(os.path.exists(output_path))

        # Check DPI of saved photo sheet
        sheet_dpi = self.check_image_dpi(output_path)
        self.assertIsNotNone(sheet_dpi, "DPI information should be present in the saved photo sheet")
        self.assertEqual(sheet_dpi[0], sheet_dpi[1], "Horizontal and vertical DPI should be the same")
        self.assertEqual(sheet_dpi[0], dpi, f"Photo sheet DPI should be {dpi}")

    def test_photo_requirements(self):
        requirements = PhotoRequirements(language='en')
        first_photo_size = self.get_first_valid_photo_size(requirements)
        self.assertIsNotNone(first_photo_size, "No valid photo size found")
        
        # Test getting resize image list
        size_info = requirements.get_resize_image_list(first_photo_size)
        self.assertIsInstance(size_info, dict)
        self.assertIn('width', size_info)
        self.assertIn('height', size_info)
        self.assertIn('electronic_size', size_info)
        self.assertIn('print_size', size_info)
        self.assertIn('resolution', size_info)

    def test_config_manager(self):
        # Initialize ConfigManager with Chinese
        config_manager = ConfigManager(language='zh')
        config_manager.load_configs()
        
        # Test Chinese configuration
        photo_size_configs_zh = config_manager.get_photo_size_configs()
        self.assertIsInstance(photo_size_configs_zh, dict)
        self.assertGreater(len(photo_size_configs_zh), 0)
        
        sheet_size_configs_zh = config_manager.get_sheet_size_configs()
        self.assertIsInstance(sheet_size_configs_zh, dict)
        self.assertGreater(len(sheet_size_configs_zh), 0)

        # Switch to English
        config_manager.switch_language('en')
        
        # Test English configuration
        photo_size_configs_en = config_manager.get_photo_size_configs()
        self.assertIsInstance(photo_size_configs_en, dict)
        self.assertGreater(len(photo_size_configs_en), 0)
        
        sheet_size_configs_en = config_manager.get_sheet_size_configs()
        self.assertIsInstance(sheet_size_configs_en, dict)
        self.assertGreater(len(sheet_size_configs_en), 0)

        # Check if some common sizes exist
        common_sizes_zh = ['一寸', '二寸 (证件照)', '五寸', '六寸']
        common_sizes_en = ['One Inch', 'Two Inch (ID Photo)', 'Five Inch', 'Six Inch']
        
        # Switch back to Chinese for checking Chinese sizes
        config_manager.switch_language('zh')
        for size_zh in common_sizes_zh:
            self.assertIn(size_zh, photo_size_configs_zh, f"Chinese config should contain {size_zh}")
        
        # Switch to English for checking English sizes
        config_manager.switch_language('en')
        for size_en in common_sizes_en:
            self.assertTrue(any(size_en.lower() in key.lower() for key in photo_size_configs_en.keys()), 
                            f"English config should contain a key with {size_en}")

        # Test language switching functionality
        self.assertEqual(config_manager.language, 'en')
        config_manager.switch_language('zh')
        self.assertEqual(config_manager.language, 'zh')

        # Test if configuration file paths are correctly updated
        self.assertTrue(config_manager.size_file.endswith('size_zh.csv'))
        self.assertTrue(config_manager.color_file.endswith('color_zh.csv'))

        # Test i18n JSON files
        i18n_dir = os.path.join(project_root, 'src', 'webui', 'i18n')
        
        with open(os.path.join(i18n_dir, 'en.json'), 'r', encoding='utf-8') as f:
            en_i18n = json.load(f)
        
        with open(os.path.join(i18n_dir, 'zh.json'), 'r', encoding='utf-8') as f:
            zh_i18n = json.load(f)
        
        # Check if i18n files have the same keys
        self.assertEqual(set(en_i18n.keys()), set(zh_i18n.keys()))
        
        # Check if at least one value is different
        different_values = [key for key in en_i18n.keys() if en_i18n[key] != zh_i18n[key]]
        self.assertGreater(len(different_values), 0, "English and Chinese i18n files should have different translations")


if __name__ == '__main__':
    unittest.main()
