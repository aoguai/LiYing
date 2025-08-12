FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 创建模型目录并声明为挂载点
RUN mkdir -p /app/src/model
VOLUME ["/app/src/model"]

RUN mkdir -p /app/output
VOLUME ["/app/output"]

# 复制项目代码
COPY . /app

# 暴露 WebUI 端口
EXPOSE 7860

# 启动 WebUI
CMD ["python", "src/webui/app.py"]
