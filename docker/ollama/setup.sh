#!/bin/sh
set -e

# 后台启动 Ollama 以便拉取模型
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to be ready..."
until ollama list > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is ready."

# 拉取 embedding 模型
EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
echo "Pulling embedding model: $EMBEDDING_MODEL"
ollama pull "$EMBEDDING_MODEL"

# 拉取 query 改写小模型
QUERY_REWRITE_MODEL="${QUERY_REWRITE_MODEL:-qwen2.5:0.5b}"
echo "Pulling query rewrite model: $QUERY_REWRITE_MODEL"
ollama pull "$QUERY_REWRITE_MODEL"

echo "All models pulled. Stopping background serve..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null || true

echo "Setup complete."
