import os
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from io import BytesIO
from pathlib import Path

import click
import mozjpeg_lossless_optimization
from PIL import Image


class QualityInteger(click.ParamType):
    name = "QualityInteger"

    @staticmethod
    def _parse_int(value):
        try:
            return int(value)
        except ValueError:
            return None

    def convert(self, value, param, ctx):
        """
        Convert a string to an integer or an integer range.

        :param value: The value to convert.
        :type value: str
        :param param: The parameter.
        :type param: XXX
        :param ctx: The context.
        :type ctx: XXX
        :return: The converted integer or integer range.
        :rtype: int or tuple[int, int]
        """
        if value.isdigit():
            return self._parse_int(value)
        parts = value.split('-')
        if len(parts) != 2:
            raise click.BadParameter(f'"{value}": The parameter does not conform to the format like 80-90 or 90')
        min_v, max_v = map(self._parse_int, parts)
        if min_v is None or max_v is None or min_v > max_v or min_v <= 0 or max_v <= 0:
            raise click.BadParameter(f'"{value}": The parameter does not conform to the format like 80-90 or 90')
        return min_v, max_v


def generate_output_path(fp, output=None):
    """
    Generate the output path.

    :param fp: The file path.
    :type fp: Path
    :param output: The output path.
    :type output: Path or None
    :return: The output path.
    :rtype: Path
    """
    uuid_str = get_uuid("".join(["AGPicCompress", str(time.time()), fp.name]))
    new_fp = Path(fp.parent, f"{fp.stem}_{uuid_str}_compressed{fp.suffix}")

    if output:
        if output.is_dir():
            # If it is a directory, check if it exists, create it if it doesn't
            output.mkdir(parents=True, exist_ok=True)
            new_fp = output / new_fp.name
        elif output.exists():
            # If it is a file, check if it exists, throw an exception if it does
            raise FileExistsError(f'"{output}": already exists')
        else:
            new_fp = output

    return new_fp


def optimize_output_path(fp, output=None, force=False):
    """
    Optimize the output path.

    :param fp: File path.
    :type fp: Path
    :param output: Output path.
    :type output: Path
    :param force: Whether to force overwrite.
    :type force: bool
    :return: Output path.
    :rtype: Path
    """
    if force:
        if output:
            if output.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                new_fp = output / fp.name
            else:
                new_fp = output
        else:
            new_fp = fp
    elif output:
        new_fp = generate_output_path(fp, output)
    else:
        new_fp = generate_output_path(fp)

    return new_fp


def find_pngquant_cmd():
    """
    Find and return the executable file path of pngquant.

    :return: The executable file path of pngquant, or None if not found.
    :rtype: str or None
    """
    pngquant_cmd = shutil.which('pngquant')
    if pngquant_cmd:
        return pngquant_cmd
    exe_extension = '.exe' if os.name == 'nt' else ''
    search_paths = [Path(__file__).resolve().parent, Path(__file__).resolve().parent / 'ext']
    for search_path in search_paths:
        pngquant_exe_path = search_path / f'pngquant{exe_extension}'
        if pngquant_exe_path.exists():
            return str(pngquant_exe_path)
    return None


def get_uuid(name):
    """
    Get the UUID string of the specified string.

    :param name: The name string for UUID generation.
    :type name: str
    :return: The UUID string generated based on the specified name.
    :rtype: str
    """
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, name))


class ImageCompressor:
    def __init__(self):
        pass

    @staticmethod
    def _save_image(img, file_path, img_format=None, quality=None, existing_bytes=None):
        """
        Save image to file, avoiding secondary compression.

        If existing_bytes is provided, it will be written directly to the file without further compression.
        Otherwise, it will use PIL's convert and BytesIO methods to get image bytes, then write to file.

        :param img: Image object. Can be None when existing_bytes is provided.
        :type img: PIL.Image.Image or None
        :param file_path: File path to save to.
        :type file_path: Path
        :param img_format: Image format ('JPEG', 'PNG', 'WEBP', etc.), must be provided when existing_bytes is None.
        :type img_format: str or None
        :param quality: Compression quality, only used when new compression is necessary.
        :type quality: int or None
        :param existing_bytes: Existing image byte data, if provided will be written directly to file.
        :type existing_bytes: bytes or None
        :return: File path
        :rtype: Path
        """
        if existing_bytes:
            with open(file_path, 'wb') as f:
                f.write(existing_bytes)
            return file_path

        if not img:
            raise ValueError("One of the img or existing_bytes parameters must be provided")

        if not img_format:
            ext = file_path.suffix.lower()
            if ext == '.jpg' or ext == '.jpeg':
                img_format = 'JPEG'
            elif ext == '.png':
                img_format = 'PNG'
            elif ext == '.webp':
                img_format = 'WEBP'
            else:
                raise ValueError(f"Unknown file format: {ext}")

        # First converted to bytes, quality parameters must be specified to avoid using the default settings of PIL
        with BytesIO() as buffer:
            save_params = {'format': img_format}
            if quality is not None:
                save_params['quality'] = quality

            img.save(buffer, **save_params)
            img_bytes = buffer.getvalue()

            with open(file_path, 'wb') as f:
                f.write(img_bytes)

        return file_path

    @staticmethod
    def compress_image(fp, force=False, quality=None, output=None, webp=False, target_size=None, size_range=None,
                       webp_quality=100):
        """
        Compression function.

        :param fp: File name or directory name.
        :type fp: Path
        :param force: Whether to overwrite if a file with the same name exists.
        :type force: bool
        :param quality: Compression quality. 80-90, or 90.
        :type quality: int or tuple[int, int]
        :param output: Output path or output directory
        :type output: Path
        :param webp: Whether to convert to WebP format.
        :type webp: bool
        :param target_size: Target file size in KB. When specified, quality is ignored.
        :type target_size: int or None
        :param size_range: A tuple of (min_size, max_size) in KB. Tries to keep quality while ensuring size is within range.
        :type size_range: tuple(int, int) or None
        :param webp_quality: Quality for WebP conversion (1-100). Default is 100.
        :type webp_quality: int
        """

        # Parameter validation
        if target_size is not None:
            if target_size <= 0:
                raise ValueError(f"Target size must be greater than 0, got {target_size}")
            if size_range is not None:
                warnings.warn("Both target_size and size_range provided. Using target_size and ignoring size_range.",
                              Warning)
                size_range = None

        if size_range is not None:
            min_size, max_size = size_range
            if min_size <= 0 or max_size <= 0:
                raise ValueError(
                    f"Size range values must be greater than 0, got min_size={min_size}, max_size={max_size}")
            if min_size >= max_size:
                raise ValueError(
                    f"Minimum size must be less than maximum size, got min_size={min_size}, max_size={max_size}")

        # Check if the file exists
        if not fp.exists():
            raise FileNotFoundError(f'"{fp}": Path or directory does not exist')
        if output:
            if not output.is_dir():
                if output.suffix == '':
                    output.mkdir(parents=True, exist_ok=True)
                elif output.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.webp']:
                    raise ValueError(f'"{output.name}": Unsupported output file format')
                elif output.suffix.lower() == '.webp':
                    webp = True
                    output = output.with_name(f"{output.stem}_2webp{fp.suffix}")
                if output.suffix.lower() != fp.suffix.lower():
                    raise ValueError('Inconsistent output file format with input file format')

        if fp.is_dir():
            for file in fp.iterdir():
                if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    ImageCompressor.compress_image(file, force, quality, output, webp, target_size, size_range,
                                                   webp_quality)
            return

        ext = fp.suffix.lower()
        if ext == '.png':
            ImageCompressor._compress_png(fp, force, quality, output, webp, target_size, size_range, webp_quality)
        elif ext in ['.jpg', '.jpeg']:
            ImageCompressor._compress_jpg(fp, force, quality, output, webp, target_size, size_range, webp_quality)
        else:
            raise ValueError(f'"{fp.name}": Unsupported output file format')

    @staticmethod
    def _convert_to_webp(fp, target_size=None, size_range=None, webp_quality=100):
        """
        Convert an image to WebP format.

        :param fp: Image file path.
        :type fp: Path
        :param target_size: Target file size in KB to maintain after conversion.
        :type target_size: int or None
        :param size_range: A tuple of (min_size, max_size) in KB to maintain after conversion.
        :type size_range: tuple(int, int) or None
        :param webp_quality: Quality for WebP conversion (1-100).
        :type webp_quality: int
        :return: WebP image file path.
        :rtype: Path or None
        """
        # Check if the file exists
        if Path(fp).exists():
            img = Image.open(fp)
            webp_fp = Path(fp).with_suffix('.webp')

            quality = webp_quality
            attempts = 0
            small_change_count = 0
            previous_size = float('inf')

            while attempts < 10:
                ImageCompressor._save_image(img, webp_fp, 'WEBP', quality)
                current_size = webp_fp.stat().st_size / 1024

                if target_size is not None:
                    if current_size <= target_size:
                        if current_size < target_size:
                            ImageCompressor._adjust_file_size(webp_fp, target_size)
                        break
                elif size_range is not None:
                    min_size, max_size = size_range
                    if current_size <= max_size:
                        if current_size < min_size:
                            ImageCompressor._adjust_file_size(webp_fp, min_size)
                        break
                else:
                    break

                size_reduction = previous_size - current_size
                if size_reduction < 5:
                    small_change_count += 1
                    if small_change_count >= 3:
                        if target_size is not None:
                            raise ValueError(
                                f"Unable to compress WebP to target size of {target_size}KB. Best achieved: {current_size:.2f}KB")
                        else:
                            raise ValueError(
                                f"Unable to compress WebP to size range of {min_size}-{max_size}KB. Best achieved: {current_size:.2f}KB")
                else:
                    small_change_count = 0

                previous_size = current_size
                if current_size > (target_size if target_size else size_range[1]):
                    quality = max(10, quality - 5)
                else:
                    quality = min(100, quality + 5)

                attempts += 1

            # Delete the original image file
            os.remove(fp)
            return webp_fp
        return None

    @staticmethod
    def _adjust_file_size(file_path, target_size_kb):
        """
        Adjust file size to match the target size by adding bytes.

        :param file_path: Path to the file to adjust
        :type file_path: Path
        :param target_size_kb: Target size in KB
        :type target_size_kb: int
        :return: True if adjustment was successful, False otherwise
        :rtype: bool
        """
        target_size_bytes = target_size_kb * 1024
        current_size_bytes = file_path.stat().st_size

        if current_size_bytes >= target_size_bytes:
            return True

        bytes_to_add = target_size_bytes - current_size_bytes

        with open(file_path, 'rb') as f:
            content = f.read()

        padding = b'\xff\xfe' + b'\x00' * (bytes_to_add - 2)

        with open(file_path, 'wb') as f:
            f.write(content + padding)

        return True

    @staticmethod
    def _compress_png(fp, force=False, quality=None, output=None, webp=False, target_size=None, size_range=None,
                      webp_quality=100):
        """
        Compress PNG images and specify compression quality.

        :param fp: Path of the image file.
        :type fp: Path
        :param force: Whether to overwrite if a file with the same name already exists, default is False.
        :type force: bool
        :param quality: Compression quality. Defaults to None.
        :type quality: int or tuple[int, int]
        :param output: Output path.
        :type output: Path
        :param webp: Whether to convert to WebP format.
        :type webp: bool
        :param target_size: Target file size in KB. When specified, quality is ignored.
        :type target_size: int or None
        :param size_range: A tuple of (min_size, max_size) in KB. Tries to keep quality while ensuring size is within range.
        :type size_range: tuple(int, int) or None
        :param webp_quality: Quality for WebP conversion (1-100). Default is 100.
        :type webp_quality: int
        """
        new_fp = optimize_output_path(fp, output, force)
        pngquant_cmd = find_pngquant_cmd()
        if not pngquant_cmd:
            raise FileNotFoundError(
                'pngquant not found. Please ensure pngquant is installed or added to the environment variable')

        adjusted_target_size = None
        adjusted_size_range = None

        if webp and (target_size is not None or size_range is not None):
            with Image.open(fp) as img:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_orig = Path(temp_dir) / f'temp_orig{fp.suffix}'
                    ImageCompressor._save_image(img, temp_orig, 'PNG')
                    orig_size = temp_orig.stat().st_size

                    temp_webp = Path(temp_dir) / 'temp.webp'
                    ImageCompressor._save_image(img, temp_webp, 'WEBP', webp_quality)
                    webp_size = temp_webp.stat().st_size

                    ratio = webp_size / orig_size if orig_size > 0 else 0.7

                    safety_factor = 1.1
                    if target_size is not None:
                        adjusted_target_size = int(target_size / ratio * safety_factor)
                    if size_range is not None:
                        min_size, max_size = size_range
                        adjusted_size_range = (
                            int(min_size / ratio * safety_factor), int(max_size / ratio * safety_factor))
        else:
            adjusted_target_size = target_size
            adjusted_size_range = size_range

        if adjusted_size_range is not None:
            min_size, max_size = adjusted_size_range
            if min_size > max_size:
                raise ValueError(f"Minimum size ({min_size}KB) cannot be greater than maximum size ({max_size}KB)")

            current_quality = quality if isinstance(quality, int) else 90
            attempts = 0
            small_change_count = 0
            previous_size = float('inf')

            while True:
                quality_command = f'--quality {current_quality}'
                command = f'{pngquant_cmd} {fp} --skip-if-larger -f -o {new_fp} {quality_command}'
                subprocess.run(command, shell=True, check=True)

                if not new_fp.exists():
                    warnings.warn(
                        f'"{fp}": The compressed image file was not generated successfully. It may no longer be compressible or no longer exist',
                        Warning)
                    return

                current_size = new_fp.stat().st_size / 1024

                if current_size <= max_size:
                    if current_size < min_size:
                        ImageCompressor._adjust_file_size(new_fp, min_size)
                    break

                size_reduction = previous_size - current_size
                if size_reduction < 5:
                    small_change_count += 1
                    if small_change_count >= 3:
                        raise ValueError(
                            f"Unable to compress image to size range of {min_size}-{max_size}KB. Best achieved: {current_size:.2f}KB")
                else:
                    small_change_count = 0

                previous_size = current_size
                current_quality = max(1, current_quality - 5)
                attempts += 1

        elif adjusted_target_size is not None:
            current_quality = 80
            attempts = 0
            small_change_count = 0
            previous_size = float('inf')

            while True:
                quality_command = f'--quality {current_quality}'
                command = f'{pngquant_cmd} {fp} --skip-if-larger -f -o {new_fp} {quality_command}'
                subprocess.run(command, shell=True, check=True)

                if not new_fp.exists():
                    warnings.warn(
                        f'"{fp}": The compressed image file was not generated successfully. It may no longer be compressible or no longer exist',
                        Warning)
                    return

                current_size = new_fp.stat().st_size / 1024

                if current_size <= adjusted_target_size:
                    ImageCompressor._adjust_file_size(new_fp, adjusted_target_size)
                    break

                size_reduction = previous_size - current_size
                if size_reduction < 5:
                    small_change_count += 1
                    if small_change_count >= 3:
                        raise ValueError(
                            f"Unable to compress image to target size of {adjusted_target_size}KB. Best achieved: {current_size:.2f}KB")
                else:
                    small_change_count = 0

                previous_size = current_size
                current_quality = max(1, current_quality - 10)
                attempts += 1
        else:
            quality_command = f'--quality {quality}' if isinstance(quality,
                                                                   int) else f'--quality {quality[0]}-{quality[1]}' if isinstance(
                quality, tuple) else ''
            command = f'{pngquant_cmd} {fp} --skip-if-larger -f -o {new_fp} {quality_command}'
            subprocess.run(command, shell=True, check=True)

            if not new_fp.exists():
                warnings.warn(
                    f'"{fp}": The compressed image file was not generated successfully. It may no longer be compressible or no longer exist',
                    Warning)
                return

        if webp:
            ImageCompressor._convert_to_webp(new_fp, target_size, size_range, webp_quality)

    @staticmethod
    def _compress_jpg(fp, force=False, quality=None, output=None, webp=False, target_size=None, size_range=None,
                      webp_quality=100):
        """
        Compress JPG images and specify compression quality.

        :param fp: Image file path.
        :type fp: Path
        :param force: Whether to overwrite if a file with the same name already exists, default is False.
        :type force: bool
        :param quality: Compression quality, default is None.
        :type quality: int or None
        :param output: Output path.
        :type output: Path
        :param webp: Whether to convert to WebP format.
        :type webp: bool
        :param target_size: Target file size in KB. When specified, quality is ignored.
        :type target_size: int or None
        :param size_range: A tuple of (min_size, max_size) in KB. Tries to keep quality while ensuring size is within range.
        :type size_range: tuple(int, int) or None
        :param webp_quality: Quality for WebP conversion (1-100). Default is 100.
        :type webp_quality: int
        """
        new_fp = optimize_output_path(fp, output, force)

        # First compress the JPEG to target size or size range
        if size_range is not None:
            min_size, max_size = size_range
            if min_size > max_size:
                raise ValueError(f"Minimum size ({min_size}KB) cannot be greater than maximum size ({max_size}KB)")

            current_quality = quality if isinstance(quality, int) else 90
            attempts = 0
            small_change_count = 0
            previous_size = float('inf')

            while True:
                with Image.open(fp) as img:
                    img = img.convert("RGB")
                    with BytesIO() as buffer:
                        img.save(buffer, format="JPEG", quality=current_quality)
                        input_jpeg_bytes = buffer.getvalue()

                optimized_jpeg_bytes = mozjpeg_lossless_optimization.optimize(input_jpeg_bytes)
                ImageCompressor._save_image(None, new_fp, existing_bytes=optimized_jpeg_bytes)

                if not new_fp.exists():
                    warnings.warn(
                        f'"{fp}": The compressed image file was not generated successfully. It may no longer be compressible or no longer exist',
                        Warning)
                    return

                current_size = new_fp.stat().st_size / 1024

                if current_size <= max_size:
                    if current_size < min_size:
                        ImageCompressor._adjust_file_size(new_fp, min_size)
                    break

                size_reduction = previous_size - current_size
                if size_reduction < 5:
                    small_change_count += 1
                    if small_change_count >= 3:
                        raise ValueError(
                            f"Unable to compress image to size range of {min_size}-{max_size}KB. Best achieved: {current_size:.2f}KB")
                else:
                    small_change_count = 0

                previous_size = current_size
                current_quality = max(1, current_quality - 5)
                attempts += 1

        elif target_size is not None:
            current_quality = 80
            attempts = 0
            small_change_count = 0
            previous_size = float('inf')

            while True:
                with Image.open(fp) as img:
                    img = img.convert("RGB")
                    with BytesIO() as buffer:
                        img.save(buffer, format="JPEG", quality=current_quality)
                        input_jpeg_bytes = buffer.getvalue()

                optimized_jpeg_bytes = mozjpeg_lossless_optimization.optimize(input_jpeg_bytes)
                ImageCompressor._save_image(None, new_fp, existing_bytes=optimized_jpeg_bytes)

                if not new_fp.exists():
                    warnings.warn(
                        f'"{fp}": The compressed image file was not generated successfully. It may no longer be compressible or no longer exist',
                        Warning)
                    return

                current_size = new_fp.stat().st_size / 1024

                if current_size <= target_size:
                    ImageCompressor._adjust_file_size(new_fp, target_size)
                    break

                size_reduction = previous_size - current_size
                if size_reduction < 5:
                    small_change_count += 1
                    if small_change_count >= 3:
                        raise ValueError(
                            f"Unable to compress image to target size of {target_size}KB. Best achieved: {current_size:.2f}KB")
                else:
                    small_change_count = 0

                previous_size = current_size
                current_quality = max(1, current_quality - 10)
                attempts += 1
        else:
            if quality is not None and not isinstance(quality, int):
                raise ValueError(f'"{quality}": Unsupported type for quality parameter')

            with Image.open(fp) as img:
                img = img.convert("RGB")
                with BytesIO() as buffer:
                    img.save(buffer, format="JPEG", quality=quality if quality else 75)
                    input_jpeg_bytes = buffer.getvalue()

            optimized_jpeg_bytes = mozjpeg_lossless_optimization.optimize(input_jpeg_bytes)
            ImageCompressor._save_image(None, new_fp, existing_bytes=optimized_jpeg_bytes)

            if not new_fp.exists():
                warnings.warn(
                    f'"{fp}": The compressed image file was not generated successfully. It may no longer be compressible or no longer exist',
                    Warning)
                return

        # If WebP conversion is requested, convert the compressed JPEG to WebP
        if webp:
            # Pass the original target_size and size_range to WebP conversion
            ImageCompressor._convert_to_webp(new_fp, target_size, size_range, webp_quality)

    @staticmethod
    def compress_image_from_bytes(image_bytes, quality=80, output_format='JPEG', webp=False, target_size=None,
                                  size_range=None, webp_quality=100):
        """
        Compresses image data and returns the compressed image data.

        :param image_bytes: The byte representation of the image data.
        :type image_bytes: bytes
        :param quality: The compression quality, ranging from 1 to 100.
        :type quality: int
        :param output_format: The output format of the image, default is 'JPEG'.
        :type output_format: str
        :param webp: Whether to convert to WebP format.
        :type webp: bool
        :param target_size: Target file size in KB. When specified, quality is ignored.
        :type target_size: int or None
        :param size_range: A tuple of (min_size, max_size) in KB. Tries to keep quality while ensuring size is within range.
        :type size_range: tuple(int, int) or None
        :param webp_quality: Quality for WebP conversion (1-100). Default is 100.
        :type webp_quality: int
        :return: The byte representation of the compressed image data.
        :rtype: bytes
        """

        # Parameter validation
        if target_size is not None:
            if target_size <= 0:
                raise ValueError(f"Target size must be greater than 0, got {target_size}")
            if size_range is not None:
                warnings.warn("Both target_size and size_range provided. Using target_size and ignoring size_range.",
                              Warning)
                size_range = None

        if size_range is not None:
            min_size, max_size = size_range
            if min_size <= 0 or max_size <= 0:
                raise ValueError(
                    f"Size range values must be greater than 0, got min_size={min_size}, max_size={max_size}")
            if min_size >= max_size:
                raise ValueError(
                    f"Minimum size must be less than maximum size, got min_size={min_size}, max_size={max_size}")

        with BytesIO(image_bytes) as img_buffer:
            img = Image.open(img_buffer).convert('RGB')

            if size_range is not None and target_size is not None:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = Path(
                        temp_dir) / f'temp_{get_uuid(f"AGPicCompress{time.time()}")}.{output_format.lower()}'
                    ImageCompressor._save_image(img, temp_file_path, output_format.upper(), quality)

                    if output_format.upper() == 'JPEG':
                        if size_range is not None:
                            ImageCompressor._compress_jpg(temp_file_path, force=True, quality=quality,
                                                          size_range=size_range, webp=webp, target_size=None,
                                                          webp_quality=webp_quality)
                        else:
                            ImageCompressor._compress_jpg(temp_file_path, force=True, target_size=target_size,
                                                          webp=webp, quality=None, webp_quality=webp_quality)
                    elif output_format.upper() == 'PNG':
                        if size_range is not None:
                            ImageCompressor._compress_png(temp_file_path, force=True, quality=quality,
                                                          size_range=size_range, webp=webp, target_size=None,
                                                          webp_quality=webp_quality)
                        else:
                            ImageCompressor._compress_png(temp_file_path, force=True, target_size=target_size,
                                                          webp=webp, quality=None, webp_quality=webp_quality)
                    else:
                        raise ValueError(f'"{output_format}": Unsupported output file format')

                    final_path = temp_file_path.with_suffix('.webp') if webp else temp_file_path

                    if final_path.exists():
                        with open(final_path, 'rb') as compressed_file:
                            compressed_img_bytes = compressed_file.read()
                            return compressed_img_bytes
                    else:
                        raise ValueError(f"Failed to generate compressed image: {final_path}")

            if output_format.upper() == 'JPEG':
                with BytesIO() as output_buffer:
                    ImageCompressor._save_image(img, Path('temp.jpg'), 'JPEG', quality)
                    with open(Path('temp.jpg'), 'rb') as temp_file:
                        compressed_img_bytes = temp_file.read()
                    os.remove(Path('temp.jpg'))

                    compressed_img_bytes = mozjpeg_lossless_optimization.optimize(compressed_img_bytes)

                    if target_size is not None:
                        with tempfile.TemporaryDirectory() as size_adjust_dir:
                            temp_file_path = Path(
                                size_adjust_dir) / f'temp_{get_uuid(f"AGPicCompress{time.time()}")}.jpg'
                            ImageCompressor._save_image(None, temp_file_path, existing_bytes=compressed_img_bytes)

                            current_size = temp_file_path.stat().st_size / 1024

                            if current_size > target_size:
                                current_quality = quality
                                attempts = 0
                                small_change_count = 0
                                previous_size = float('inf')

                                while current_size > target_size and attempts < 10:
                                    current_quality = max(1, current_quality - 10)

                                    with Image.open(temp_file_path) as img:
                                        img = img.convert("RGB")
                                        with BytesIO() as buffer:
                                            img.save(buffer, format="JPEG", quality=current_quality)
                                            input_jpeg_bytes = buffer.getvalue()

                                    optimized_jpeg_bytes = mozjpeg_lossless_optimization.optimize(input_jpeg_bytes)
                                    ImageCompressor._save_image(None, temp_file_path,
                                                                existing_bytes=optimized_jpeg_bytes)

                                    current_size = temp_file_path.stat().st_size / 1024

                                    size_reduction = previous_size - current_size
                                    if size_reduction < 5:
                                        small_change_count += 1
                                        if small_change_count >= 3:
                                            raise ValueError(
                                                f"Unable to compress image to target size of {target_size}KB. Best achieved: {current_size:.2f}KB")
                                    else:
                                        small_change_count = 0

                                    previous_size = current_size
                                    attempts += 1

                            if current_size < target_size:
                                ImageCompressor._adjust_file_size(temp_file_path, target_size)
                                with open(temp_file_path, 'rb') as adjusted_file:
                                    compressed_img_bytes = adjusted_file.read()
                            # Ensure the temporary file is read back if it was used for size adjustment
                            elif temp_file_path.exists():  # Check if temp_file_path was actually used and exists
                                with open(temp_file_path, 'rb') as adjusted_file:
                                    compressed_img_bytes = adjusted_file.read()

                    elif size_range is not None:
                        min_size, max_size = size_range
                        with tempfile.TemporaryDirectory() as size_adjust_dir:
                            temp_file_path = Path(
                                size_adjust_dir) / f'temp_{get_uuid(f"AGPicCompress{time.time()}")}.jpg'
                            ImageCompressor._save_image(None, temp_file_path, existing_bytes=compressed_img_bytes)

                            current_size = temp_file_path.stat().st_size / 1024

                            if current_size > max_size:
                                current_quality = quality
                                attempts = 0
                                small_change_count = 0
                                previous_size = float('inf')

                                while current_size > max_size and attempts < 10:
                                    current_quality = max(1, current_quality - 10)

                                    with Image.open(temp_file_path) as img:
                                        img = img.convert("RGB")
                                        with BytesIO() as buffer:
                                            img.save(buffer, format="JPEG", quality=current_quality)
                                            input_jpeg_bytes = buffer.getvalue()

                                    optimized_jpeg_bytes = mozjpeg_lossless_optimization.optimize(input_jpeg_bytes)
                                    ImageCompressor._save_image(None, temp_file_path,
                                                                existing_bytes=optimized_jpeg_bytes)

                                    current_size = temp_file_path.stat().st_size / 1024

                                    size_reduction = previous_size - current_size
                                    if size_reduction < 5:
                                        small_change_count += 1
                                        if small_change_count >= 3:
                                            raise ValueError(
                                                f"Unable to compress image to size range of {min_size}-{max_size}KB. Best achieved: {current_size:.2f}KB")
                                    else:
                                        small_change_count = 0

                                    previous_size = current_size
                                    attempts += 1

                            if current_size < min_size:
                                ImageCompressor._adjust_file_size(temp_file_path, min_size)
                                with open(temp_file_path, 'rb') as adjusted_file:
                                    compressed_img_bytes = adjusted_file.read()
                            # Ensure the temporary file is read back if it was used for size adjustment
                            elif temp_file_path.exists():  # Check if temp_file_path was actually used and exists
                                with open(temp_file_path, 'rb') as adjusted_file:
                                    compressed_img_bytes = adjusted_file.read()

                    # Add WebP conversion here if webp is True
                    if webp:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_jpg_path = Path(temp_dir) / f'temp_input_for_webp_{get_uuid(str(time.time()))}.jpg'
                            with open(temp_jpg_path, 'wb') as f_temp_jpg:
                                f_temp_jpg.write(compressed_img_bytes)

                            # Call _convert_to_webp with original target_size and size_range
                            webp_converted_path = ImageCompressor._convert_to_webp(
                                temp_jpg_path,
                                target_size,  # Pass original target_size
                                size_range,  # Pass original size_range
                                webp_quality
                            )

                            if webp_converted_path and webp_converted_path.exists():
                                with open(webp_converted_path, 'rb') as f_webp:
                                    compressed_img_bytes = f_webp.read()
                                # _convert_to_webp might have deleted temp_jpg_path, so no explicit deletion here for it
                            else:
                                warnings.warn(
                                    f"Failed to convert JPEG to WebP. Original JPEG bytes will be returned.",
                                    Warning
                                )
                                # compressed_img_bytes remains the JPEG bytes

            elif output_format.upper() == 'PNG':
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_png_file_path = Path(temp_dir) / f'temp_{get_uuid(f"AGPicCompress{time.time()}")}.png'

                    ImageCompressor._save_image(None, temp_png_file_path, existing_bytes=image_bytes)

                    new_fp = optimize_output_path(temp_png_file_path, Path(temp_dir), False)
                    pngquant_cmd = find_pngquant_cmd()
                    if not pngquant_cmd:
                        raise FileNotFoundError(
                            'pngquant not found. Please ensure pngquant is installed or added to the environment variable')
                    quality_command = f'--quality {quality}' if isinstance(quality,
                                                                           int) else f'--quality {quality[0]}-{quality[1]}' if isinstance(
                        quality, tuple) else ''
                    command = f'{pngquant_cmd} {temp_png_file_path} --skip-if-larger -f -o {new_fp} {quality_command}'
                    subprocess.run(command, shell=True, check=True)
                    if new_fp.exists():
                        with open(new_fp, 'rb') as compressed_img_file:
                            compressed_img_bytes = compressed_img_file.read()

                            if target_size is not None:
                                with tempfile.TemporaryDirectory() as size_adjust_dir:
                                    temp_file_path = Path(
                                        size_adjust_dir) / f'temp_{get_uuid(f"AGPicCompress{time.time()}")}.png'
                                    ImageCompressor._save_image(None, temp_file_path,
                                                                existing_bytes=compressed_img_bytes)

                                    current_size = temp_file_path.stat().st_size / 1024

                                    if current_size > target_size:
                                        current_quality = quality
                                        attempts = 0
                                        small_change_count = 0
                                        previous_size = float('inf')

                                        while current_size > target_size and attempts < 10:
                                            current_quality = max(1, current_quality - 10)

                                            quality_command = f'--quality {current_quality}'
                                            command = f'{pngquant_cmd} {temp_file_path} --skip-if-larger -f -o {temp_file_path} {quality_command}'
                                            subprocess.run(command, shell=True, check=True)

                                            if not temp_file_path.exists():
                                                break

                                            current_size = temp_file_path.stat().st_size / 1024

                                            size_reduction = previous_size - current_size
                                            if size_reduction < 5:
                                                small_change_count += 1
                                                if small_change_count >= 3:
                                                    raise ValueError(
                                                        f"Unable to compress image to target size of {target_size}KB. Best achieved: {current_size:.2f}KB")
                                            else:
                                                small_change_count = 0

                                            previous_size = current_size
                                            attempts += 1

                                    if current_size < target_size:
                                        ImageCompressor._adjust_file_size(temp_file_path, target_size)
                                        with open(temp_file_path, 'rb') as adjusted_file:
                                            compressed_img_bytes = adjusted_file.read()

                            elif size_range is not None:
                                min_size, max_size = size_range
                                with tempfile.TemporaryDirectory() as size_adjust_dir:
                                    temp_file_path = Path(
                                        size_adjust_dir) / f'temp_{get_uuid(f"AGPicCompress{time.time()}")}.png'

                                    ImageCompressor._save_image(None, temp_file_path,
                                                                existing_bytes=compressed_img_bytes)

                                    current_size = temp_file_path.stat().st_size / 1024

                                    if current_size > max_size:
                                        current_quality = quality
                                        attempts = 0
                                        small_change_count = 0
                                        previous_size = float('inf')

                                        while current_size > max_size and attempts < 10:
                                            current_quality = max(1, current_quality - 10)

                                            quality_command = f'--quality {current_quality}'
                                            command = f'{pngquant_cmd} {temp_file_path} --skip-if-larger -f -o {temp_file_path} {quality_command}'
                                            subprocess.run(command, shell=True, check=True)

                                            if not temp_file_path.exists():
                                                break

                                            current_size = temp_file_path.stat().st_size / 1024

                                            size_reduction = previous_size - current_size
                                            if size_reduction < 5:
                                                small_change_count += 1
                                                if small_change_count >= 3:
                                                    raise ValueError(
                                                        f"Unable to compress image to size range of {min_size}-{max_size}KB. Best achieved: {current_size:.2f}KB")
                                            else:
                                                small_change_count = 0

                                            previous_size = current_size
                                            attempts += 1

                                    if current_size < min_size:
                                        ImageCompressor._adjust_file_size(temp_file_path, min_size)
                                        with open(temp_file_path, 'rb') as adjusted_file:
                                            compressed_img_bytes = adjusted_file.read()
                    else:
                        warnings.warn(
                            'The compressed image file was not generated successfully. It may no longer be compressible or no longer exist',
                            Warning)
                        return None

                    if webp:
                        with tempfile.TemporaryDirectory() as webp_temp_dir:
                            temp_img_path = Path(webp_temp_dir) / f'temp_{get_uuid(f"AGPicCompress{time.time()}")}.png'

                            ImageCompressor._save_image(None, temp_img_path, existing_bytes=compressed_img_bytes)

                            webp_path = ImageCompressor._convert_to_webp(temp_img_path, target_size, size_range,
                                                                         webp_quality)

                            if webp_path and webp_path.exists():
                                with open(webp_path, 'rb') as webp_file:
                                    compressed_img_bytes = webp_file.read()
                            else:
                                with tempfile.TemporaryDirectory() as webp_out_dir:
                                    temp_webp_path = Path(webp_out_dir) / 'temp.webp'
                                    img = Image.open(BytesIO(compressed_img_bytes))
                                    ImageCompressor._save_image(img, temp_webp_path, 'WEBP', webp_quality)
                                    with open(temp_webp_path, 'rb') as webp_file:
                                        compressed_img_bytes = webp_file.read()

                                if size_range is not None:
                                    min_size, max_size = size_range
                                    with tempfile.TemporaryDirectory() as size_adjust_dir:
                                        temp_file_path = Path(
                                            size_adjust_dir) / f'temp_{get_uuid(f"AGPicCompress{time.time()}")}.webp'

                                        ImageCompressor._save_image(None, temp_file_path,
                                                                    existing_bytes=compressed_img_bytes)

                                        current_size = temp_file_path.stat().st_size / 1024

                                        if current_size < min_size:
                                            ImageCompressor._adjust_file_size(temp_file_path, min_size)
                                            with open(temp_file_path, 'rb') as adjusted_file:
                                                compressed_img_bytes = adjusted_file.read()
            else:
                raise ValueError(f'"{output_format}": Unsupported output file format')
        return compressed_img_bytes

    @staticmethod
    @click.command()
    @click.argument('fp')
    @click.option(
        "--force", "-f", "--violent",
        is_flag=True,
        help="Whether to overwrite if a file with the same name exists, defaults to False."
    )
    @click.option('--quality', "-q", default="80", type=QualityInteger(),
                  help="Compression quality. 80-90, or 90, default is 80.")
    @click.option('--output', '-o', help='Output path or output directory.')
    @click.option('--webp', is_flag=True, help='Convert images to WebP format, default is False.')
    @click.option('--target-size', '-t', type=int, help='Target file size in KB. When specified, quality is ignored.')
    @click.option('--size-range', '-s', nargs=2, type=int,
                  help='Min and max size in KB. Tries to maintain quality while ensuring size is within range.')
    @click.option('--webp-quality', '-wq', type=int, default=100,
                  help='Quality for WebP conversion (1-100). Default is 100.')
    def cli_compress(fp, force=False, quality=None, output=None, webp=False, target_size=None, size_range=None,
                     webp_quality=100):
        """
        Compress images via command line.

        :param fp: Image file path or directory path.
        :type fp: str

        :param force: Whether to overwrite if a file with the same name exists, defaults to False.
        :type force: bool

        :param quality: Compression quality. 80-90, or 90, default is 80.
        :type quality: int or tuple[int, int]

        :param output: Output path or output directory.
        :type output: str

        :param webp: Convert images to WebP format, default is False.
        :type webp: bool

        :param target_size: Target file size in KB. When specified, quality is ignored.
        :type target_size: int or None

        :param size_range: Min and max size in KB. Tries to maintain quality while ensuring size is within range.
        :type size_range: tuple(int, int) or None

        :param webp_quality: Quality for WebP conversion (1-100). Default is 100.
        :type webp_quality: int
        """
        if not fp:
            raise ValueError(f'"{fp}": The file path or directory cannot be empty')

        fp_path = Path(fp)

        if output:
            output_path = Path(output) if len(output) > 0 else None

        size_range_tuple = tuple(size_range) if size_range else None

        ImageCompressor.compress_image(fp_path, force, quality, output_path, webp, target_size, size_range_tuple,
                                       webp_quality)
        return
