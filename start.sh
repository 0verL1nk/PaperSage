#!/bin/bash
set -e

RQ_WORKER_COUNT=${RQ_WORKER_COUNT:-2}
WORKER_PIDS=()

# 清理函数：当收到退出信号时，清理所有后台进程
cleanup() {
    echo "正在关闭服务..."
    for worker_pid in "${WORKER_PIDS[@]}"; do
        kill "$worker_pid" 2>/dev/null || true
    done
    kill "$REDIS_PID" 2>/dev/null || true
    for worker_pid in "${WORKER_PIDS[@]}"; do
        wait "$worker_pid" 2>/dev/null || true
    done
    wait "$REDIS_PID" 2>/dev/null || true
    exit 0
}

# 注册清理函数
trap cleanup SIGTERM SIGINT

# 启动 Redis 服务器（后台运行）
echo "启动 Redis 服务器..."
redis-server --daemonize yes --protected-mode no
REDIS_PID=$(pgrep -f "redis-server" | head -1)

# 等待 Redis 启动
echo "等待 Redis 启动..."
for i in {1..10}; do
    if redis-cli ping > /dev/null 2>&1; then
        echo "Redis 已启动"
        break
    fi
    sleep 1
done

# 启动多个 RQ worker（后台运行）
echo "启动 RQ worker，数量: ${RQ_WORKER_COUNT}"
cd /app
for i in $(seq 1 "$RQ_WORKER_COUNT"); do
    rq worker tasks --url redis://localhost:6379/0 > "/tmp/rq_worker_${i}.log" 2>&1 &
    WORKER_PIDS+=("$!")
done

# 等待 worker 启动
sleep 2

# 启动 Streamlit 应用（前台运行，这样容器不会退出）
echo "启动 Streamlit 应用..."
exec streamlit run main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
