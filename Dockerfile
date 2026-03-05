# 使用官方Python镜像作为基础镜像
FROM python:3.11-slim AS base

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# 配置使用国内 apt 源（清华大学镜像）
# 检测 Debian 版本并配置对应的镜像源
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        # Debian 12+ 使用新的 sources 格式
        sed -i 's|https://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
        sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        # Debian 11 及更早版本或 Ubuntu
        sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
        sed -i 's|https://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
        sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
        sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
    fi

# 更新 apt 包列表（独立层，便于缓存）
RUN apt-get update

# 安装基础工具和健康检查工具
RUN apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Redis 服务器
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 复制项目配置文件（先复制配置文件以利用缓存层）
COPY pyproject.toml ./

# 使用 uv 从 pyproject.toml 编译生成 requirements.txt（独立层，便于缓存）
# 使用 BuildKit 缓存挂载，即使 pyproject.toml 改变，已下载的包也会被缓存
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip compile pyproject.toml -o /tmp/requirements.txt

# 安装 Python 依赖（独立层，便于缓存）
# 即使 requirements.txt 改变，已下载的包也会被缓存
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p uploads

# 暴露Streamlit默认端口
EXPOSE 8501

# 复制启动脚本
COPY start.sh /start.sh
RUN chmod +x /start.sh

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 使用启动脚本启动所有服务
CMD ["/start.sh"]

