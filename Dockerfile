# 福彩推荐系统 Docker 镜像
# 构建: docker build -t lottery-web .
# 运行: docker run -d -p 8000:8080 -e LOTTERY_PASS=your-password -v ${PWD}/data:/app/data lottery-web
#
# DevCloud 部署注意:
#   1. LOTTERY_PASS 必须在 DevCloud 应用配置中设为运行时环境变量（不是 build-arg）
#   2. 端口必须为 8080（平台硬性要求）
#   3. DEBUG=false 以启用 cookie Secure 标志
#   4. CORS_ORIGINS 设置为 DevCloud 分配的域名

FROM python:3.11-slim

WORKDIR /app

# 安装依赖（DevCloud 预处理器会自动注入阿里云 pip 镜像，无需手动配置）
COPY requirements-min.txt ./
RUN pip install --no-cache-dir -r requirements-min.txt

# 复制项目文件
COPY src/ src/
COPY config/ config/
COPY static/ static/
COPY scripts/ scripts/

# 创建所有数据目录和非 root 用户
RUN mkdir -p data/ssq data/sd data/3d data/qlc data/kl8 data/users model output && \
    groupadd -r lottery && useradd -r -g lottery -d /app lottery && \
    chown -R lottery:lottery /app

# 以非 root 用户运行
USER lottery

# 暴露端口（DevCloud 仅暴露 8080）
EXPOSE 8080

# 健康检查（start-period 120s：首次启动需下载数据，给足时间）
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]