"""
TaskIQ Broker 설정 (Redis)
"""
import os
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
from taskiq import TaskiqEvents

# Redis URL (환경 변수 또는 기본값)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Result Backend 설정 (작업 결과 저장)
result_backend = RedisAsyncResultBackend(
    redis_url=REDIS_URL,
)

# Broker 설정 (작업 큐)
broker = ListQueueBroker(
    url=REDIS_URL,
    result_backend=result_backend,
    # BRPOP은 작업이 들어올 때까지 대기하는 명령이므로 읽기 타임아웃을 두지 않는다.
    socket_timeout=None,
)

