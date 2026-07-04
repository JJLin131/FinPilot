FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（某些 sentence-transformers 版本需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app.py .

EXPOSE 8765

ENV CROSSENCODER_PORT=8765
ENV RERANKER_MODEL=BAAI/bge-reranker-v2-m3
ENV CROSSENCODER_DEVICE=cpu
ENV CROSSENCODER_MAX_LENGTH=512

CMD ["python", "app.py"]
