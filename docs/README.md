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

如果你是 Windows 用户且没有代码阅览需求，可以[下载整合包](https://github.com/aoguai/LiYing/releases/latest)（已在 Windows 7 SP1 &  Windows 10 测试），解压将图片或目录拖入 run_zh.bat 即可启动 LiYing。

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

注： 如果您使用的是 Windows 7 系统请您至少需要是 Windows 7 SP1 以上版本，且要求 `onnxruntime==1.14.0`

#### 下载对应模型

您需要下载该项目使用到的模型并将其放置在 `LiYing/src/model` 中。或者您可以在 CIL 中指定模型路径。

| 用途                     | 模型名称              | 下载链接                                                                                                             | 来源                                                     |
|------------------------|--------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| 人脸识别                  | Yunnet             | [下载链接](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | [Yunnet](https://github.com/ShiqiYu/libfacedetection)  |
| 主体识别替换背景              | RMBG-1.4           | [下载链接](https://huggingface.co/briaai/RMBG-1.4/blob/main/onnx/model.onnx)                                           | [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)     |
| 人体识别                  | yolov8n-pose       | [下载链接](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)                           | [ultralytics](https://github.com/ultralytics/ultralytics) |

**注： 对于 yolov8n-pose 模型，您需要将其导出为 ONNX 模型，您可以参考[官方文档](https://docs.ultralytics.com/integrations/onnx/)实现**

#### 运行

```shell
cd LiYing/src
python main.py --help
```

对于 Window 用户，项目提供了 bat 运行脚本方便您使用:

```shell
cd LiYing
run.bat ./images/test1.jpg
```

#### CIL 参数信息与帮助
```shell
python main.py --help
Usage: main.py [OPTIONS] IMG_PATH

Options:
  -y, --yolov8-model-path PATH    YOLOv8 模型路径
  -u, --yunet-model-path PATH     YuNet 模型路径
  -r, --rmbg-model-path PATH      RMBG 模型路径
  -b, --rgb-list RGB_LIST         RGB 通道值列表（英文逗号分隔），用于图像合成
  -s, --save-path PATH            保存路径
  -p, --photo-type TEXT           照片类型
  --photo-sheet-size TEXT         选择照片表格的尺寸
  -c, --compress / --no-compress  是否压缩图像
  -sc, --save-corrected / --no-save-corrected
                                  是否保存修正图像后的图片
  -bg, --change-background / --no-change-background
                                  是否替换背景
  -sb, --save-background / --no-save-background
                                  是否保存替换背景后的图像
  -sr, --sheet-rows INTEGER       照片表格的行数
  -sc, --sheet-cols INTEGER       照片表格的列数
  --rotate / --no-rotate          是否旋转照片90度
  -rs, --resize / --no-resize     是否调整图像尺寸
  -srz, --save-resized / --no-save-resized
                                  是否保存调整尺寸后的图像
  --help                          Show this message and exit.

```

#### 其他配置

在该版本中，在`data/data-zh.ini`中设置了常规的证件照配置，您可以在`photo-type`和`photo-sheet-size`参数中使用。

同时你可以修改该配置文件，自定义证件照类型。针对中文环境，其格式为
```text
[XXX]
打印尺寸 = XXXcm x XXXcm
电子版尺寸 = XXXpx x XXXpx
分辨率 = XXXdpi
```
其中节名称及`[XXX]`和`电子版尺寸 = XXXpx x XXXpx`是必须的。

其中节名称代表了其`photo-type`和`photo-sheet-size`参数输入值。

同时`photo-type`和`photo-sheet-size`还支持直接输入形如`XXXpx x XXXpx`的字符串，代表宽高。

## 更新日志

- **2024/08/06 更新**
  - 新增 photo-type 和 photo-sheet-size 支持直接输入宽高像素，支持使用 data.ini 配置
  - 修复 部分 i18n 导致的已知问题，现在可以兼容中英文配置
  - 修复 其他已知BUG

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
