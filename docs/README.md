# LiYing

简体中文 | [English](./README-EN.md)

LiYing 是一套适用于自动化完成一般照相馆后期流程的照片自动处理的程序。

## 介绍

LiYing 可以完成人体、人脸自动识别，角度自动纠正，自动更换任意背景色，任意尺寸证件照自动裁切，并自动排版。

LiYing 可以完全离线运行。所有图像处理操作都在本地运行。

### 简单工作流说明

![workflows](../images/workflows.png)

### 效果展示

| ![test1](../images/test1.jpg) | ![test2](../images/test2.jpg) | ![test3](../images/test3.jpg) |
| ----------------------------- | ---------------------------- | ---------------------------- |
| ![test1_output_sheet](../images/test1_output_sheet.jpg)(1寸-5寸相片纸-3*3) | ![test2_output_sheet](../images/test2_output_sheet.jpg)(2寸-5寸相片纸-2*2) | ![test3_output_sheet](../images/test3_output_sheet.jpg)(1寸-6寸相片纸-4*2) |

**注：本项目仅针对证件照图像处理，而非要求任意照片图像都可以完美执行，所以该项目的输入图片应该是符合一般要求的单人肖像照片。**

**如果您使用复杂图片制作证件照出现意外情况属于正常现象。**

## 开始使用

### 整合包

如果你是 Windows 用户且没有代码阅览需求，可以[下载整合包](https://github.com/aoguai/LiYing/releases/latest)（已在 Windows 7 SP1 &  Windows 10 测试）

整合包从未包含模型，您可以参考 [下载对应模型](https://github.com/aoguai/LiYing?tab=readme-ov-file#%E4%B8%8B%E8%BD%BD%E5%AF%B9%E5%BA%94%E6%A8%A1%E5%9E%8B) 章节说明来下载模型并放入正确的位置

同时如果运行存在问题，请先尝试按照 [先决条件](https://github.com/aoguai/LiYing?tab=readme-ov-file#%E5%85%88%E5%86%B3%E6%9D%A1%E4%BB%B6) 章节完善环境，如果没问题可以忽略

#### 运行整合包

运行 BAT 脚本
```shell
cd LiYing
run.bat ./images/test1.jpg
```

运行 WebUI 界面
```shell
# 运行 WebUI
cd LiYing
run_webui.bat
# 浏览器访问 127.0.0.1:7860
```

### 设置和安装

您可以按照以下说明进行安装和配置，从而在本地环境中使用 LiYing。

#### 先决条件

LiYing 依赖于 AGPicCompress ，而 AGPicCompress 需要依赖于 mozjpeg 和 pngquant

其中你可能需要手动安装 pngquant，你可以参考 [pngquant 官方文档](https://pngquant.org/)并将其添加到对应位置

LiYing 会在以下位置检测 pngquant 是否存在，你可以自由配置
- 环境变量（推荐）
- LiYing/src 目录下
- LiYing/src 目录下的 `ext` 目录

以便 AGPicCompress 能够找到 pngquant 并使用它进行 PNG 图片的压缩。

#### Microsoft Visual C++ Redistributable 依赖

您需要安装最新 [Microsoft Visual C++ Redistributable 依赖](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)


如果您使用的是 Windows 系统，您的最低版本应该是 Windows 7 SP1 及以上。

#### 从源码构建

您可以通过以下方式获取 LiYing 项目的代码：

```shell
git clone https://github.com/aoguai/LiYing
cd LiYing ## 进入 LiYing 目录
pip install -r requirements.txt # install Python helpers' dependencies
```

**注： 如果您使用的是 Windows 7 系统请您至少需要是 Windows 7 SP1 以上版本，且要求 `onnxruntime==1.14.0, orjson==3.10.7, gradio==4.44.1`**

#### 下载对应模型

您需要下载该项目使用到的模型并将其放置在 `LiYing/src/model` 中。或者您可以在 CIL 中指定模型路径。

| 用途                     | 模型名称              | 下载链接                                                                                                             | 来源                                                     |
|------------------------|--------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| 人脸识别                  | Yunnet             | [下载链接](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | [Yunnet](https://github.com/ShiqiYu/libfacedetection)  |
| 主体识别替换背景              | RMBG-1.4           | [下载链接](https://huggingface.co/briaai/RMBG-1.4/blob/main/onnx/model.onnx)                                           | [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)     |
| 人体识别                  | yolov8n-pose       | [下载链接](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)                           | [ultralytics](https://github.com/ultralytics/ultralytics) |

**注： 对于 yolov8n-pose 模型，您需要将其导出为 ONNX 模型，您可以参考[官方文档](https://docs.ultralytics.com/integrations/onnx/)实现**

同时，我们提供了转换好的 ONNX 模型，您可以直接下载使用：

| 下载方式         | 链接                                                                             |
|--------------|--------------------------------------------------------------------------------|
| Google Drive | [下载链接](https://drive.google.com/file/d/1F8EQfwkeq4s-P2W4xQjD28c4rxPuX1R3/view) |
| 百度网盘         | [下载链接(提取码：ahr9)](https://pan.baidu.com/s/1QhzW53vCbhkIzvrncRqJow?pwd=ahr9)             |
| Github releases | [下载链接](https://github.com/aoguai/LiYing/releases/latest)             |

#### 运行

```shell
# 查看 CIL 帮助
cd LiYing/src
python main.py --help
```

对于 Window 用户，项目提供了 bat 运行脚本方便您使用:

```shell
# 运行 BAT 脚本
cd LiYing
run.bat ./images/test1.jpg
```

```shell
# 运行 WebUI
cd LiYing/src/webui
python app.py
```

#### CIL 参数信息与帮助
```shell
python main.py --help 
Usage: main.py [OPTIONS] IMG_PATH

Options:
  -y, --yolov8-model-path PATH    YOLOv8 模型路径
  -u, --yunet-model-path PATH     YuNet 模型路径
  -r, --rmbg-model-path PATH      RMBG 模型路径
  -sz, --size-config PATH         尺寸配置文件路径
  -cl, --color-config PATH        颜色配置文件路径
  -b, --rgb-list RGB_LIST         RGB 通道值列表（英文逗号分隔），用于图像合成
  -s, --save-path PATH            保存路径
  -p, --photo-type TEXT           照片类型
  -ps, --photo-sheet-size TEXT    选择照片表格的尺寸
  -c, --compress / --no-compress  是否压缩图像（使用 AGPicCompress 压缩）
  -sv, --save-corrected / --no-save-corrected
                                  是否保存修正图像后的图片
  -bg, --change-background / --no-change-background
                                  是否替换背景
  -sb, --save-background / --no-save-background
                                  是否保存替换背景后的图像
  -lo, --layout-only              仅排版照片，不更换背景
  -sr, --sheet-rows INTEGER       照片表格的行数
  -sc, --sheet-cols INTEGER       照片表格的列数
  -rt, --rotate / --no-rotate     是否旋转照片90度
  -rs, --resize / --no-resize     是否调整图像尺寸
  -svr, --save-resized / --no-save-resized
                                  是否保存调整尺寸后的图像
  -al, --add-crop-lines / --no-add-crop-lines
                                  在照片表格上添加裁剪线
  -ts, --target-size INTEGER      目标文件大小（KB）。指定后将忽略质量和大小范围参数。
  -szr, --size-range SIZE_RANGE   文件大小范围（KB），格式为最小值,最大值（例如：10,20）
  -uc, --use-csv-size / --no-use-csv-size
                                  是否使用CSV中的文件大小限制
  --help                          Show this message and exit.

```

#### 配置文件

在该版本中，在`data`目录中设置了常规的证件照配置`size_XX.csv`与常用颜色配置`color_XX.csv`，您可以自行按照给出的 CSV 模板格式修改或增删配置。

## 更新日志

**注意该版本对 CIL 参数进行了更改，为了避免问题请你仔细阅读最新 CIL 帮助文档**

- **2025/02/07 更新**
  - **添加 WebUI**
  - 优化 配置方式，用 CSV 替换 INI 配置
  - 添加 CI/CD 方便自动构建与测试
  - 添加 仅排版照片, 是否在照片表格上添加裁剪线 选项
  - 完善 对非脸部图像的兜底处理
  - 修复 已知BUG
  - 添加修正补充了更多尺寸
<details> 
    <summary>往期更新日志</summary>

- **2024/08/06 更新**
  - 新增 photo-type 和 photo-sheet-size 支持直接输入宽高像素，支持使用 data.ini 配置
  - 修复 部分 i18n 导致的已知问题，现在可以兼容中英文配置
  - 修复 其他已知BUG
</details>

## 致谢

该项目的制作初衷和项目名称来源于帮助我的父母更轻松的完成他们的工作，在此感谢我的父母。

### 相关

同时特别感谢以下项目和贡献者：

提供模型与理论

- [Yunnet](https://github.com/ShiqiYu/libfacedetection)
- [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)
- [ultralytics](https://github.com/ultralytics/ultralytics)

或许你会对图片压缩部分感兴趣，那是我另一个开源项目

- [AGPicCompress](https://github.com/aoguai/AGPicCompress)

它依赖于

- [mozjpeg](https://github.com/mozilla/mozjpeg)
- [pngquant](https://github.com/kornelski/pngquant)
- [mozjpeg-lossless-optimization](https://github.com/wanadev/mozjpeg-lossless-optimization)

## 贡献

LiYing 是一个开源项目，非常欢迎社区的参与。要为该项目做出贡献，请遵循[贡献指南](./CONTRIBUTING.md)。

## License 说明

[LiYing](https://github.com/aoguai/LiYing) 使用 AGPL-3.0 license 进行开源，详情请参阅 [LICENSE](../LICENSE) 文件。

## Star History

<a href="https://star-history.com/#aoguai/LiYing&Timeline">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=aoguai/subscription&type=Timeline&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=aoguai/subscription&type=Timeline" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=aoguai/subscription&type=Timeline" />
  </picture>
</a>
