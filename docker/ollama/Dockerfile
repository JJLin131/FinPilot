FROM ollama/ollama:latest

# 启动脚本：等待 ollama serve 就绪后拉取模型
COPY setup.sh /setup.sh
RUN chmod +x /setup.sh

# 覆盖入口点：先后台启动 serve，拉模型，然后前台保持 serve
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["/setup.sh && exec ollama serve"]
