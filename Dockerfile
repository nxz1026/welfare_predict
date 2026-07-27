# 福彩推荐系统 Docker 镜像
# 构建: docker build -t lottery-web .
# 运行: docker run -d -p 8000:8000 -v ${PWD}/data:/app/data lottery-web

FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目
COPY . .

# 创建数据目录
RUN mkdir -p data/ssq data/3d data/qlc data/users model output

# 环境变量
ENV LOTTERY_USER=admin
ENV LOTTERY_PASS=caipiao2026
ENV LOTTERY_SECRET=change-me-in-production

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
