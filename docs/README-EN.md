<div align="center">

<h1>LiYing</h1>

[简体中文](./README.md) | English

[![GitHub release](https://img.shields.io/github/v/release/aoguai/LiYing?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/aoguai/LiYing/releases/latest)
[![GitHub stars](https://img.shields.io/github/stars/aoguai/LiYing?color=ffcb47&labelColor=black&style=flat-square)](https://github.com/aoguai/LiYing/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/aoguai/LiYing?color=ff80eb&labelColor=black&style=flat-square)](https://github.com/aoguai/LiYing/issues)
[![GitHub contributors](https://img.shields.io/github/contributors/aoguai/LiYing?color=c4f042&labelColor=black&style=flat-square)](https://github.com/aoguai/LiYing/graphs/contributors)
[![GitHub forks](https://img.shields.io/github/forks/aoguai/LiYing?color=8ae8ff&labelColor=black&style=flat-square)](https://github.com/aoguai/LiYing/network/members)
[![License](https://img.shields.io/badge/license-AGPL--3.0-white?labelColor=black&style=flat-square)](../LICENSE)

<p>LiYing is an automated photo processing program designed for automating the post-processing workflow of ID photos in general photo studios.</p>

</div>

<br>

## 🧭 Project Introduction

LiYing can automatically identify human bodies and faces, correct angles, change background colors, crop passport photos to any size, and automatically arrange them.

LiYing can run completely offline. All image processing operations are performed locally.

### Workflow

![workflows](../images/workflows.png)

### Showcase

| ![test1](../images/test1.jpg) | ![test2](../images/test2.jpg) | ![test3](../images/test3.jpg) |
| ----------------------------- | ---------------------------- | ---------------------------- |
| ![test1_output_sheet](../images/test1_output_sheet.jpg)(1-inch on 5-inch photo paper - 3x3) | ![test2_output_sheet](../images/test2_output_sheet.jpg)(2-inch on 5-inch photo paper - 2x2) | ![test3_output_sheet](../images/test3_output_sheet.jpg)(1-inch on 6-inch photo paper - 4x2) |

**Note: This project is specifically for processing passport photos and may not work perfectly on any arbitrary image. The input images should be standard single-person portrait photos.**

**It is normal for unexpected results to occur if you use complex images to create passport photos.**

<br>

## Getting Started

### Bundled Package

If you are a Windows user and do not need to review the code, you can [download the bundled package](https://github.com/aoguai/LiYing/releases/latest) (tested on Windows 7 SP1 & Windows 10).

The bundled package does not include any models. You can refer to the [Downloading the Required Models](#downloading-the-required-models) section for instructions on downloading the models and placing them in the correct directory.

If you encounter issues while running the program, please first check the [Prerequisites](#prerequisites) section to ensure your environment is properly set up. If everything is fine, you can ignore this step.

#### Running the bundled package

Run the BAT script:
```shell
cd LiYing
run.bat ./images/test1.jpg
```

Run the WebUI interface:
```shell
# Run WebUI
cd LiYing
run_webui.bat
# Open your browser and visit 127.0.0.1:7860
```

### 🛠 Prerequisites

1. **Dependencies**
   - LiYing depends on AGPicCompress
   - AGPicCompress requires `mozjpeg` and `pngquant`
   - You may need to manually install `pngquant`, refer to the [pngquant official documentation](https://pngquant.org/)

2. **pngquant Configuration Location**
   - Environment variables (recommended)
   - LiYing/src directory
   - `ext` directory under LiYing/src

3. **System Requirements**
   - Windows users need to install the latest [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
   - If you are using Windows, your minimum version should be Windows 7 SP1 or higher.

### 🧪 Building from Source

1. Clone the project:

```shell
git clone https://github.com/aoguai/LiYing
cd LiYing ## Enter the LiYing directory
pip install -r requirements.txt # Install Python helpers' dependencies
```

**Note: If you are using Windows 7, ensure you have at least Windows 7 SP1 and `onnxruntime==1.14.0, orjson==3.10.7, gradio==4.44.1`.**

### GPU-Accelerated Inference (Optional)

To leverage an NVIDIA GPU for accelerated inference, proceed with the following measures:

1.  Ensure that both the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and the [cuDNN library]((https://developer.nvidia.com/cudnn)) are correctly installed on your system.
2.  [Consult the official compatibility matrix to determine the required versions for ONNX Runtime, CUDA, and cuDNN that correspond with one another.](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
3.  Install the GPU-enabled build of the ONNX Runtime library:
    ```bash
    # First, uninstall the CPU-only variant if it is currently installed.
    pip uninstall onnxruntime
    # Install the GPU-enabled version, ensuring its compatibility with your environment.
    pip install onnxruntime-gpu
    ```

The current version of the system is engineered to automatically detect the presence of a compatible GPU. Upon detection, it will prioritize the GPU for inference operations, seamlessly reverting to the CPU in its absence. This functionality requires no additional configuration.

**Should any complications arise, it is imperative to first verify the mutual compatibility between your installed versions of Python, CUDA, cuDNN, and `onnxruntime-gpu`.**

<br>

### 📦 Downloading the Required Models

Download the models used by the project and place them in `LiYing/src/model`, or specify the model paths in the command line.

| Purpose                   | Model Name        | Download Link                                                                                                                                           | Source                                                     |
|---------------------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| Face Recognition          | Yunnet            | [Download Link](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)                           | [Yunnet](https://github.com/ShiqiYu/libfacedetection)      |
| Subject Recognition and Background Replacement | RMBG-1.4/2.0 | [1.4 Download Link](https://huggingface.co/briaai/RMBG-1.4/blob/main/onnx/model.onnx)/[2.0 Download Link](https://huggingface.co/briaai/RMBG-2.0/tree/main/onnx) | [BRIA AI](https://huggingface.co/briaai)         |
| Body Recognition          | yolov8n-pose      | [Download Link](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)                                                         | [ultralytics](https://github.com/ultralytics/ultralytics) |

**Note: For the yolov8n-pose model, you need to export it to an ONNX model. Refer to the [official documentation](https://docs.ultralytics.com/integrations/onnx/) for instructions.**

We also provide pre-converted ONNX models that you can download and use directly:

| Download Method  | Link  |
|-----------------|--------------------------------------------------------------------------------|
| Google Drive   | [Download Link](https://drive.google.com/file/d/1F8EQfwkeq4s-P2W4xQjD28c4rxPuX1R3/view) |
| Baidu Netdisk  | [Download Link (Extraction Code: ahr9)](https://pan.baidu.com/s/1QhzW53vCbhkIzvrncRqJow?pwd=ahr9) |
| GitHub Releases | [Download Link](https://github.com/aoguai/LiYing/releases/latest) |

### 🚀 Running

View CIL help:
```shell
cd LiYing/src
python main.py --help
```

For Windows users, the project provides a batch script for convenience:

```shell
# Run BAT script
cd LiYing
run.bat ./images/test1.jpg
```

Run WebUI:
```shell
cd LiYing/src/webui
python app.py
```

### 🧾 CLI Parameters and Help

```shell
python main.py --help
Usage: main.py [OPTIONS] IMG_PATH

Options:
  -y, --yolov8-model-path PATH    Path to YOLOv8 model
  -u, --yunet-model-path PATH     Path to YuNet model
  -r, --rmbg-model-path PATH      Path to RMBG model
  -sz, --size-config PATH         Path to size configuration file
  -cl, --color-config PATH        Path to color configuration file
  -b, --rgb-list RGB_LIST         RGB channel values list (comma-separated)
                                  for image composition
  -s, --save-path PATH            Path to save the output image
  -p, --photo-type TEXT           Photo types
  -ps, --photo-sheet-size TEXT    Size of the photo sheet
  -c, --compress / --no-compress  Whether to compress the image
  -sv, --save-corrected / --no-save-corrected
                                  Whether to save the corrected image
  -bg, --change-background / --no-change-background
                                  Whether to change the background
  -sb, --save-background / --no-save-background
                                  Whether to save the image with changed
                                  background
  -lo, --layout-only              Only layout the photo without changing
                                  background
  -sr, --sheet-rows INTEGER       Number of rows in the photo sheet
  -sc, --sheet-cols INTEGER       Number of columns in the photo sheet
  -rt, --rotate / --no-rotate     Whether to rotate the photo by 90 degrees
  -rs, --resize / --no-resize     Whether to resize the image
  -svr, --save-resized / --no-save-resized
                                  Whether to save the resized image
  -al, --add-crop-lines / --no-add-crop-lines
                                  Add crop lines to the photo sheet
  -ts, --target-size INTEGER      Target file size in KB. When specified,
                                  ignores quality and size-range.
  -szr, --size-range SIZE_RANGE   File size range in KB as min,max (e.g.,
                                  10,20)
  -uc, --use-csv-size / --no-use-csv-size
                                  Whether to use file size limits from CSV
  -lp, --layout-position INTEGER RANGE
                                  Layout position (0-8): 0=top-left, 1=top,
                                  2=top-right, 3=middle-left, 4=center,
                                  5=middle-right, 6=bottom-left, 7=bottom,
                                  8=bottom-right  [0<=x<=8]
  --help                          Show this message and exit.
```

### 🗂 Configuration Files

In this version, the `data` directory contains standard ID photo configuration files (`size_XX.csv`) and commonly used color configurations (`color_XX.csv`). You can modify, add, or remove configurations based on the provided CSV template format.

<br>

### 🐳 Docker Deployment

---

#### ️1. Build the Image

##### Build with docker-compose

Run the following command in the project root directory:

```bash
docker compose build
````

##### Manually build the image

Run the following command in the project root directory:

```bash
docker build -t liying/webui:latest .
```

---

### 2. Start the Service

Start the Gradio Web UI service with the following command:

```bash
docker compose up -d
```

Once started, open your browser and visit:

```
http://127.0.0.1:7860
```

---

If you encounter any issues, please first verify that at least one model file is placed in `src/model/` and ensure the port is not already in use.

For more details or advanced configuration, check the [`Dockerfile`](./Dockerfile) and [`docker-compose.yml`](./docker-compose.yml).

<br>

## 🧱 Changelog

**Note: This version includes changes to CIL parameters. Please carefully read the latest CIL help documentation to avoid issues.**

- **2025/06/30 Update**
  - Added `size_range` option, allowing users to input a min and max file size, attempting to maintain quality while keeping the file size within the range.
  - Added `target_size` option to control the photo file size.
  - Added support for RMBG-2.0 and higher iterations of yolov8 (requires Latest environment).
  - Added automatic builds for CLI/BAT/WEBUI versions.
  - Added model path configuration options.
  - Fixed known bugs.

<details>
    <summary>Previous Changelog</summary>

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

<br>

## 🙏 Acknowledgments

The project was created to help my parents complete their work more easily. I would like to thank my parents for their support.

### Related Projects

Special thanks to the following projects and contributors for providing models and theories:

- [Yunnet](https://github.com/ShiqiYu/libfacedetection)
- [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)
- [ultralytics](https://github.com/ultralytics/ultralytics)

You might also be interested in the image compression part, which is another open-source project of mine:

- [AGPicCompress](https://github.com/aoguai/AGPicCompress)

It depends on:

- [mozjpeg](https://github.com/mozilla/mozjpeg)
- [pngquant](https://github.com/kornelski/pngquant)
- [mozjpeg-lossless-optimization](https://github.com/wanadev/mozjpeg-lossless-optimization)

<br>

## 🤝 Contribution

LiYing is an open-source project, and community participation is highly welcomed. To contribute to this project, please follow the [Contribution Guide](./CONTRIBUTING.md).

<br>

## 📄 License Notice

[LiYing](https://github.com/aoguai/LiYing) is open-sourced under the AGPL-3.0 license. For details, please refer to the [LICENSE](../LICENSE) file.

<br>

## 💖 Sponsors

If this project is helpful to you, feel free to give any appreciation, it helps me a lot, thank you for your support!

```
USDT(TRON):TWFDp8aZMWZHPXjBodyhfPeK8LUyrWe9mi
```

<img src="../images/usdt_thanks.jpg" width = "300" height = "300" alt="usdt_thanks"/>

<br>

## ⭐ Star History

<a href="https://star-history.com/#aoguai/LiYing&Timeline">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=aoguai/subscription&type=Timeline&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=aoguai/subscription&type=Timeline" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=aoguai/subscription&type=Timeline" />
  </picture>
</a>
