# 福彩推荐系统 Docker 镜像
# 构建: docker build --build-arg LOTTERY_PASS=your-password -t lottery-web .
# 运行: docker run -d -p 8000:8000 -v ${PWD}/data:/app/data lottery-web

FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目（排除 .git, __pycache__, tests/, docs/ 等）
COPY src/ src/
COPY config/ config/
COPY static/ static/
COPY scripts/ scripts/
COPY requirements.txt .
COPY .env.example .env.example 2>/dev/null || true
COPY Makefile .

# 创建数据目录
RUN mkdir -p data/ssq data/3d data/qlc data/users model output

# 环境变量（必须通过 --build-arg 或 .env 文件注入，禁止硬编码！）
ARG LOTTERY_USER=admin
ARG LOTTERY_PASS=${LOTTERY_PASS:?LOTTERY_PASS must be set via --build-arg}
ARG LOTTERY_SECRET=${LOTTERY_SECRET:-auto-generated-default}

ENV LOTTERY_USER=$LOTTERY_USER
ENV LOTTERY_PASS=$LOTTERY_PASS
ENV LOTTERY_SECRET=$LOTTERY_SECRET

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
