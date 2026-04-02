"""
Task backend for distributed task processing.

这个模块把“任务提交”与“任务执行”拆成了统一抽象，调用方只需要提交
一个可序列化的任务字典，不需要关心任务最终是立即执行，还是先落库后由
独立 worker 异步消费。

核心实现原理：
1. `TaskBackend` 定义统一接口，屏蔽不同运行模式的差异。
2. `SyncTaskBackend` 直接把任务交给 executor，适合测试或单进程场景。
3. `BrokerTaskBackend` 只负责把任务持久化到 PostgreSQL 的
   `async_operations` 表，把数据库当作 broker；真正的领取、轮询、
   执行由外部 worker 完成。

这样设计的价值是 API 层可以稳定地复用同一套提交逻辑，而部署方式可以在
“本地立即执行”和“生产异步处理”之间切换，而无需改调用代码。
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


def fq_table(table: str, schema: str | None = None) -> str:
    """
    Get fully-qualified table name with optional schema prefix.

    实现原理：多租户场景下，同一张逻辑表可能位于不同 schema。
    这里统一拼接 schema 前缀，避免调用方手写 SQL 表名时重复处理。
    """
    if schema:
        return f'"{schema}".{table}'
    return table


class TaskBackend(ABC):
    """
    Abstract base class for task execution backends.

    Implementations must:
    1. Store/publish task events (as serializable dicts)
    2. Execute tasks through a provided executor callback (optional)

    The backend treats tasks as pure dictionaries that can be serialized
    and stored in the database. The executor (typically MemoryEngine.execute_task)
    receives the dict and routes it to the appropriate handler.

    实现原理：这里采用“提交通道”和“执行器”分离的设计。
    - backend 决定任务如何被投递，例如立即执行或落库排队。
    - executor 决定任务被取出后如何真正执行业务逻辑。

    这样可以把基础设施层的差异与业务执行逻辑解耦，避免 MemoryEngine
    直接依赖某一种具体的队列/数据库实现。
    """

    def __init__(self):
        """Initialize the task backend."""
        self._executor: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._initialized = False

    def set_executor(self, executor: Callable[[dict[str, Any]], Awaitable[None]]):
        """
        Set the executor callback for processing tasks.

        Args:
            executor: Async function that takes a task dict and executes it

        实现原理：backend 本身不包含具体业务处理代码，而是通过注入的
        executor 回调完成执行。这种反向依赖让 backend 可以保持通用，
        同时允许上层按运行环境决定具体执行入口。
        """
        self._executor = executor

    @abstractmethod
    async def initialize(self):
        """
        Initialize the backend (e.g., connect to database).
        """
        pass

    @abstractmethod
    async def submit_task(self, task_dict: dict[str, Any]):
        """
        Submit a task for execution.

        Args:
            task_dict: Task as a dictionary (must be serializable)
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """
        Shutdown the backend gracefully.
        """
        pass

    async def _execute_task(self, task_dict: dict[str, Any]):
        """
        Execute a task through the registered executor.

        Args:
            task_dict: Task dictionary to execute

        实现原理：这是所有 backend 共用的最终执行入口。
        具体 backend 只需要决定“什么时候调用它”，而异常捕获和日志记录
        放在这里统一处理，避免不同 backend 各自复制一套执行保护逻辑。
        """
        if self._executor is None:
            # 没有注册 executor 时直接跳过，避免在测试或半初始化阶段抛出
            # 级联异常；日志里保留 task_type 方便定位是哪类任务被丢弃。
            task_type = task_dict.get("type", "unknown")
            logger.warning(f"No executor registered, skipping task {task_type}")
            return

        try:
            await self._executor(task_dict)
        except Exception as e:
            # 这里吞掉异常并记录日志，是为了避免后台任务执行失败向上冒泡后
            # 破坏调度循环。生产环境里失败任务通常由上层状态机/worker 重试
            # 或标记失败，而不是让整个 backend 崩掉。
            task_type = task_dict.get("type", "unknown")
            logger.error(f"Error executing task {task_type}: {e}")
            import traceback

            traceback.print_exc()


class SyncTaskBackend(TaskBackend):
    """
    Synchronous task backend that executes tasks immediately.

    This is useful for tests and embedded/CLI usage where we don't want
    background workers. Tasks are executed inline rather than being queued.

    实现原理：把“提交任务”退化为“立刻调用 executor”，省去 broker、
    worker、状态轮询这些异步基础设施，因此适合单进程测试、CLI 或嵌入式
    场景。
    """

    async def initialize(self):
        """No-op for sync backend."""
        self._initialized = True
        logger.debug("SyncTaskBackend initialized")

    async def submit_task(self, task_dict: dict[str, Any]):
        """
        Execute the task immediately (synchronously).

        Args:
            task_dict: Task dictionary to execute

        实现原理：第一次提交时自动初始化，然后直接走 `_execute_task()`。
        因为没有中间队列，调用方提交成功基本就等价于任务已经进入执行阶段。
        """
        if not self._initialized:
            await self.initialize()

        await self._execute_task(task_dict)

    async def shutdown(self):
        """No-op for sync backend."""
        self._initialized = False
        logger.debug("SyncTaskBackend shutdown")


class BrokerTaskBackend(TaskBackend):
    """
    Task backend using PostgreSQL as broker.

    submit_task() stores task_payload in async_operations table.
    Actual polling and execution is handled separately by WorkerPoller.

    This backend is used by the API to store tasks. Workers poll
    the database separately to claim and execute tasks.

    实现原理：这里并不直接执行任务，而是把任务 JSON 持久化到数据库。
    数据库表承担了“消息队列/任务表”的角色：
    - API 进程负责写入任务。
    - Worker 进程负责扫描 pending 任务、抢占、执行并更新状态。

    这种设计比引入额外消息队列更简单，适合当前系统已经强依赖 PostgreSQL
    的场景，同时任务状态也能天然落在同一份持久化数据里，便于追踪和审计。
    """

    def __init__(
        self,
        pool_getter: Callable[[], "asyncpg.Pool"],
        schema: str | None = None,
        schema_getter: Callable[[], str | None] | None = None,
    ):
        """
        Initialize the broker task backend.

        Args:
            pool_getter: Callable that returns the asyncpg connection pool
            schema: Database schema for multi-tenant support (optional, static)
            schema_getter: Callable that returns current schema dynamically (optional).
                          If set, takes precedence over static schema for submit_task.

        实现原理：`pool_getter` 和 `schema_getter` 都采用延迟获取，而不是在
        构造时就固定依赖对象。这样可以兼容应用启动顺序、多租户上下文切换，
        以及测试中动态替换连接池/Schema 的场景。
        """
        super().__init__()
        self._pool_getter = pool_getter
        self._schema = schema
        self._schema_getter = schema_getter

    async def initialize(self):
        """Initialize the backend."""
        self._initialized = True
        logger.info("BrokerTaskBackend initialized")

    async def submit_task(self, task_dict: dict[str, Any]):
        """
        Store task payload in async_operations table.

        The task_dict should contain an 'operation_id' if updating an existing
        operation record, otherwise a new operation will be created.

        Args:
            task_dict: Task dictionary to store (must be JSON serializable)

        实现原理：
        1. 任务先被编码成 JSON，作为 `task_payload` 落库。
        2. 如果已有 `operation_id`，说明外部已经预创建了操作记录，此时只更新
           载荷，让后续 worker 按既有 operation 继续流转。
        3. 如果没有 `operation_id`，backend 自己补建一条 pending 记录，
           让“先有任务、后有 operation”这类场景也能复用同一张表。

        这样一张 `async_operations` 表同时承担了任务队列和任务状态表的职责。
        """
        if not self._initialized:
            await self.initialize()

        pool = self._pool_getter()
        operation_id = task_dict.get("operation_id")
        task_type = task_dict.get("type", "unknown")
        bank_id = task_dict.get("bank_id")

        # task payload 最终要以 JSONB 写入 PostgreSQL，但部分任务参数里会带
        # datetime；这里统一转成 ISO 8601 字符串，保证跨进程、跨语言读取时
        # 语义稳定，也避免 json.dumps 直接失败。
        from datetime import datetime

        def datetime_encoder(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        payload_json = json.dumps(task_dict, default=datetime_encoder)

        schema = self._schema_getter() if self._schema_getter else self._schema
        table = fq_table("async_operations", schema)

        if operation_id:
            # 已有 operation_id 时只更新载荷，不重建记录。
            # 这样可以保留之前已经写入的关联元数据和生命周期字段。
            await pool.execute(
                f"""
                UPDATE {table}
                SET task_payload = $1::jsonb, updated_at = now()
                WHERE operation_id = $2
                """,
                payload_json,
                operation_id,
            )
            logger.debug(f"Updated task payload for operation {operation_id}")
        else:
            # 某些轻量任务不会提前创建 operation 记录，这里补插入一条 pending
            # 记录，让 worker 仍然通过统一表结构消费任务。
            import uuid

            new_id = uuid.uuid4()
            await pool.execute(
                f"""
                INSERT INTO {table} (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, $3, 'pending', $4::jsonb)
                """,
                new_id,
                bank_id,
                task_type,
                payload_json,
            )
            logger.debug(f"Created new operation {new_id} for task type {task_type}")

    async def shutdown(self):
        """Shutdown the backend."""
        self._initialized = False
        logger.info("BrokerTaskBackend shutdown")

    async def wait_for_pending_tasks(self, timeout: float = 120.0):
        """
        Wait for pending tasks to be processed.

        In the broker model, this polls the database to check if tasks
        for this process have been completed. This is useful in tests
        when worker_enabled=True (API processes its own tasks).

        Args:
            timeout: Maximum time to wait in seconds

        实现原理：测试环境里如果 API 自己也启动了 worker，提交任务后往往需要
        等待异步消费完成。这里通过轮询 `async_operations` 表里仍处于 pending
        且带有 `task_payload` 的记录数，来判断是否还有待处理任务。

        这不是高精度同步原语，而是一个简单、稳妥的“最终完成”探测机制：
        不依赖进程内事件对象，能够跨 API/worker 边界工作，因此更适合当前
        broker-based 架构下的集成测试。
        """
        import asyncio

        pool = self._pool_getter()
        schema = self._schema_getter() if self._schema_getter else self._schema
        table = fq_table("async_operations", schema)

        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            # 只要还有 pending + task_payload 的记录，就说明仍存在尚未被 worker
            # 完整处理的任务；轮询间隔固定为 0.5s，在测试里足够简单可靠。
            count = await pool.fetchval(
                f"""
                SELECT COUNT(*) FROM {table}
                WHERE status = 'pending' AND task_payload IS NOT NULL
                """
            )

            if count == 0:
                return

            await asyncio.sleep(0.5)

        logger.warning(f"Timeout waiting for pending tasks after {timeout}s")
