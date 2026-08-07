# 福彩推荐系统 Docker 镜像
# 构建: docker build --build-arg LOTTERY_PASS=your-password -t lottery-web .
# 运行: docker run -d -p 8000:8080 -v ${PWD}/data:/app/data lottery-web

FROM python:3.11-slim

WORKDIR /app

# 安装依赖
# 设置 INSTALL_TF=1 以安装 TensorFlow（默认跳过，使用 requirements-min.txt）
ARG INSTALL_TF=0
COPY requirements-min.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-min.txt
RUN if [ "$INSTALL_TF" = "1" ]; then \
        pip install --no-cache-dir tensorflow==2.15.1 keras==2.15.0; \
    fi

# 复制项目（排除 .git, __pycache__, tests/, docs/ 等）
COPY src/ src/
COPY config/ config/
COPY static/ static/
COPY scripts/ scripts/
COPY .env.example .env.example 2>/dev/null || true
COPY Makefile .

# 创建数据目录和非 root 用户
RUN mkdir -p data/ssq data/3d data/qlc data/users model output && \
    groupadd -r lottery && useradd -r -g lottery -d /app lottery && \
    chown -R lottery:lottery /app

# 环境变量（必须通过 --build-arg 或 .env 文件注入，禁止硬编码！）
ARG LOTTERY_USER=admin
ARG LOTTERY_PASS  # 必须通过 --build-arg 设置，无默认值

ENV LOTTERY_USER=$LOTTERY_USER
ENV LOTTERY_PASS=$LOTTERY_PASS

# 以非 root 用户运行
USER lottery

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# 启动命令（DevCloud 仅暴露 8080 端口）
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
