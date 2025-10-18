FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    "onnxruntime>=1.17.0" \
    "orjson>=3.9.15" \
    "gradio>=4.19.2" \
    -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "src/webui/app.py", "--server_name", "0.0.0.0", "--server_port", "7860", "--deployment_mode", "server"]
