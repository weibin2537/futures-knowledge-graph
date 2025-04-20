# backend/Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 使用中国镜像源
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建上传文件夹
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]