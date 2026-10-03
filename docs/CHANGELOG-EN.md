# Changelog

[简体中文](./CHANGELOG.md)

**Note: This version includes changes to CIL parameters. Please carefully read the latest CIL help documentation to avoid issues.**

- **2026/10/03 Update**
  - Added aspect-ratio cropping with optional photo sheet generation (#18).
  - Added face-height and top-margin controls for face-aware composition in CLI, BAT, and WebUI workflows (#22).
  - Added ONNX-based portrait skin retouching and whitening in CLI, BAT, and WebUI, with independently adjustable retouching and whitening intensity.
  - Added optional RetinaFace face detection for the photo layout pipeline.
  - Fixed YOLOv8 preprocessing for RGBA PNG images with an alpha channel.
  - Fixed image orientation and rotation/cropping issues by unifying EXIF-aware image loading and detection inputs, and correcting shoulder keypoint mapping and crop coordinate calculations (#21).
  - Fixed YuNet face bounding box coordinate conversion for portrait composition.
  - Fixed garbled text caused by GBK encoding in the Chinese BAT launcher.
  - Added the `libgl1` runtime dependency to Docker (#20).

<details>
    <summary>Previous Changelog</summary>

- **2026/02/16 Update**
  - Added Docker deployment support.
  - Added GPU-accelerated inference support.
  - Added `photos-spacing` option.
  - Added `layout-position` option.
  - Added support for transparent background output and fast background preview.
  - Added batch upload/processing and batch downloads for WebUI.
  - Optimized WebUI image download for server deployment.
  - Fixed other known bugs.

- **2025/06/30 Update**
  - Added `size_range` option, allowing users to input a min and max file size, attempting to maintain quality while keeping the file size within the range.
  - Added `target_size` option to control the photo file size.
  - Added support for RMBG-2.0 and higher iterations of yolov8 (requires Latest environment).
  - Added automatic builds for CLI/BAT/WEBUI versions.
  - Added model path configuration options.
  - Fixed known bugs.

- **2025/02/07 Update**
  - **Added WebUI**
  - Optimized configuration method by replacing INI files with CSV
  - Added CI/CD for automated builds and testing
  - Added options for layout-only photos and whether to add crop lines on the photo grid
  - Improved fallback handling for non-face images
  - Fixed known bugs
  - Added and refined more photo sizes

- **2024/08/06 Update**
  - Added support for entering width and height in pixels directly for `photo-type` and `photo-sheet-size`, and support for configuration via `data.ini`.
  - Fixed issues related to some i18n configurations; now compatible with both English and Chinese settings.
  - Fixed other known bugs.

</details>
