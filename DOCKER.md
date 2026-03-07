# Docker 部署指南

## 🐳 使用 Dockerfile 构建镜像

### 方式一：使用 Docker Compose（推荐）

1. **设置环境变量**
   ```bash
   export DASHSCOPE_API_KEY=your_api_key_here
   ```

2. **启动服务**
   ```bash
   docker-compose up -d
   ```
   说明：
   - Compose 会同时启动 `app` + `mineru-api` 两个服务。
   - `app` 默认设置 `DOC_PARSE_BACKEND=mineru`，PDF 优先走 MinerU API 解析。
   - 直接本地 `streamlit run main.py` 不启用 MinerU，仍走本地解析链路。
   - 默认 MinerU 镜像为 `mineru:latest`，可在 `.env` 中通过 `MINERU_IMAGE` 覆盖。

3. **查看日志**
   ```bash
   docker-compose logs -f app
   ```

4. **停止服务**
   ```bash
   docker-compose down
   ```

5. **访问应用**
   打开浏览器访问 `http://localhost:8501`

### 方式二：使用 Docker 命令

1. **构建镜像**
   ```bash
   docker build -t llm-app:latest .
   ```

2. **运行容器**
   ```bash
   docker run -d \
     --name llm-app \
     -p 8501:8501 \
     -e DASHSCOPE_API_KEY=your_api_key_here \
     -v $(pwd)/database.sqlite:/app/database.sqlite \
     -v $(pwd)/uploads:/app/uploads \
     llm-app:latest
   ```

## 📋 Dockerfile 优化说明

### 优化点

1. **分层缓存优化**
   - 先复制 `requirements.txt` 并安装依赖，代码变更时不会重新安装依赖
   - 减少构建时间

2. **系统依赖最小化**
   - 使用 `python:3.11-slim` 基础镜像
   - 只安装 textract 必需的系统依赖
   - 使用 `--no-install-recommends` 减少镜像大小
   - 安装后清理 apt 缓存

3. **符合项目要求**
   - 使用 `pip==24.0`（符合README要求）
   - 设置 `PYTHONUNBUFFERED=1` 确保日志实时输出

4. **健康检查**
   - 添加 Streamlit 健康检查端点
   - 便于容器编排工具监控

5. **数据持久化**
   - 通过 volume 挂载数据库和上传文件目录
   - 确保数据不丢失

## 🔧 环境变量配置

| 变量名 | 说明 | 默认值 | 必需 |
|--------|------|--------|------|
| `DASHSCOPE_API_KEY` | 通义千问API密钥 | - | ✅ |

## 🚀 生产环境建议

1. **使用多阶段构建进一步优化镜像大小**（可选）
2. **配置反向代理**（如 Nginx）处理 HTTPS
3. **使用 Docker Secrets 管理敏感信息**
4. **配置日志收集**（如 ELK Stack）
5. **设置资源限制**（CPU、内存）

## 📝 注意事项

- 首次运行会自动创建 SQLite 数据库文件（包含 tokens 表）
- `uploads` 目录会在容器内自动创建
- Token 管理已内置在 SQLite 数据库中，无需额外服务
- 建议使用 `docker-compose` 方式部署
