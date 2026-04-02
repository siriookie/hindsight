"""
Worker poller for distributed task execution.

Polls PostgreSQL for pending tasks and executes them using
FOR UPDATE SKIP LOCKED for safe concurrent claiming.
"""

import asyncio
import json
import logging
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .exceptions import RetryTaskAt

if TYPE_CHECKING:
    import asyncpg

    from hindsight_api.extensions.tenant import TenantExtension

logger = logging.getLogger(__name__)

# Progress logging interval in seconds
PROGRESS_LOG_INTERVAL = 30


def fq_table(table: str, schema: str | None = None) -> str:
    """Get fully-qualified table name with optional schema prefix."""
    if schema:
        return f'"{schema}".{table}'
    return table


@dataclass
class ClaimedTask:
    """A task claimed from the database with its schema context."""

    operation_id: str
    task_dict: dict[str, Any]
    schema: str | None

class WorkerPoller:
    """
    Polls PostgreSQL for pending tasks and executes them.

    Uses FOR UPDATE SKIP LOCKED for safe distributed claiming,
    allowing multiple workers to process tasks without conflicts.

    Supports dynamic multi-tenant discovery via tenant_extension.

    实现原理：`WorkerPoller` 是 broker 模式里的真正消费者。
    上游只负责把任务写入 `async_operations.task_payload`，而 poller 负责
    轮询数据库、原子 claim、并发执行、失败重试和状态回写。

    因为“写入任务”和“执行任务”被拆开，系统既可以跑独立 worker 进程，
    也可以在 API 进程内嵌 worker，并且可以按租户 schema 动态发现任务。
    """

    def __init__(
        self,
        pool: "asyncpg.Pool",
        worker_id: str,
        executor: Callable[[dict[str, Any]], Awaitable[None]],
        poll_interval_ms: int = 500,
        schema: str | None = None,
        tenant_extension: "TenantExtension | None" = None,
        max_slots: int = 10,
        consolidation_max_slots: int = 2,
    ):
        """
        Initialize the worker poller.

        Args:
            pool: asyncpg connection pool
            worker_id: Unique identifier for this worker
            executor: Async function to execute tasks (typically MemoryEngine.execute_task)
            poll_interval_ms: Interval between polls when no tasks found (milliseconds)
            schema: Database schema for single-tenant support (deprecated, use tenant_extension)
            tenant_extension: Extension for dynamic multi-tenant discovery. If None, creates a
                            DefaultTenantExtension with the configured schema.
            max_slots: Maximum concurrent tasks per worker
            consolidation_max_slots: Maximum concurrent consolidation tasks per worker
        """
        self._pool = pool
        self._worker_id = worker_id
        self._executor = executor
        self._poll_interval_ms = poll_interval_ms
        self._schema = schema
        # Always set tenant extension (use DefaultTenantExtension if none provided)
        if tenant_extension is None:
            from ..extensions.builtin.tenant import DefaultTenantExtension

            # Pass schema parameter to DefaultTenantExtension if explicitly provided
            config = {"schema": schema} if schema else {}
            tenant_extension = DefaultTenantExtension(config=config)
        self._tenant_extension = tenant_extension
        self._max_slots = max_slots
        self._consolidation_max_slots = consolidation_max_slots
        self._shutdown = asyncio.Event()
        self._current_tasks: set[asyncio.Task] = set()
        self._in_flight_count = 0
        self._in_flight_lock = asyncio.Lock()
        self._last_progress_log = 0.0
        self._tasks_completed_since_log = 0
        # Track active tasks locally: operation_id -> (op_type, bank_id, schema, asyncio.Task)
        self._active_tasks: dict[str, tuple[str, str, str | None, asyncio.Task]] = {}
        # Track in-flight tasks by operation type
        self._in_flight_by_type: dict[str, int] = {}

    async def _get_schemas(self) -> list[str | None]:
        """
        Get list of schemas to poll. Returns [None] for default schema (no prefix).

        实现原理：多租户模式下，任务不会都落在同一个 schema。
        这里每轮都动态读取 tenant_extension 返回的租户列表，而不是在启动时
        固定下来，这样新增租户后无需重启 worker，也能被后续轮询覆盖。
        """
        from ..config import DEFAULT_DATABASE_SCHEMA

        tenants = await self._tenant_extension.list_tenants()
        # Convert default schema to None for SQL compatibility (no prefix), keep others as-is
        return [t.schema if t.schema != DEFAULT_DATABASE_SCHEMA else None for t in tenants]

    async def _get_available_slots(self) -> tuple[int, int]:
        """
        Calculate available slots for claiming tasks.

        Returns:
            (total_available, consolidation_available) tuple

        实现原理：claim 之前先按本地 in-flight 状态计算“还能再领多少任务”，
        避免 worker 一次把太多任务改成 `processing`，但本进程并没有足够
        执行能力，导致任务长时间堆在 processing 状态。

        这里把普通任务和 consolidation 分开限流，是因为 consolidation
        通常更重、更耗资源，而且同 bank 的 consolidation 还带有串行约束。
        """
        async with self._in_flight_lock:
            total_in_flight = self._in_flight_count
            consolidation_in_flight = self._in_flight_by_type.get("consolidation", 0)

        total_available = max(0, self._max_slots - total_in_flight)
        consolidation_available = max(0, self._consolidation_max_slots - consolidation_in_flight)

        return total_available, consolidation_available

    async def wait_for_active_tasks(self, timeout: float = 10.0) -> bool:
        """
        Wait for all active background tasks to complete (test helper).

        This is a test-only utility that allows tests to synchronize with
        fire-and-forget background tasks without using sleep().

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all tasks completed, False if timeout was reached
        """
        start_time = asyncio.get_event_loop().time()
        while True:
            async with self._in_flight_lock:
                if self._in_flight_count == 0:
                    return True

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return False

            # Short sleep to avoid busy-waiting
            await asyncio.sleep(0.01)

    async def claim_batch(self) -> list[ClaimedTask]:
        """
        Claim pending tasks atomically across all tenant schemas,
        respecting slot limits (total and consolidation).

        Uses FOR UPDATE SKIP LOCKED to ensure no conflicts with other workers.

        Returns:
            List of ClaimedTask objects containing operation_id, task_dict, and schema

        实现原理：这一层只负责“跨 schema 的调度编排”，不直接写 claim SQL。
        它先拿到当前 worker 的剩余并发额度，再按 schema 逐个尝试 claim：
        - `remaining_total` 控制本 worker 的总并发上限
        - `remaining_consolidation` 控制重任务的独立额度

        这样做的价值是把“跨租户分发”和“单 schema 原子 claim”拆开，
        从而保证一轮轮询里不会因为前面某个 schema 领太多任务而突破全局
        slot 限制。
        """
        # 先按本地执行状态计算还能认领多少任务；如果没有空位，直接返回，
        # 避免无意义地访问数据库。
        total_available, consolidation_available = await self._get_available_slots()

        if total_available <= 0:
            return []

        schemas = await self._get_schemas()
        all_tasks: list[ClaimedTask] = []
        remaining_total = total_available
        remaining_consolidation = consolidation_available

        for schema in schemas:
            if remaining_total <= 0:
                break

            # 单个 schema 内的 claim 由子函数在事务里完成；这里负责把当前轮
            # 剩余额度传下去，确保跨 schema 总量不会超领。
            tasks = await self._claim_batch_for_schema(schema, remaining_total, remaining_consolidation)

            # consolidation 有独立上限，因此要根据本轮实际领到的任务类型回扣
            # 对应额度；否则后续 schema 仍可能继续领到过多 consolidation。
            for task in tasks:
                op_type = task.task_dict.get("operation_type", "unknown")
                if op_type == "consolidation":
                    remaining_consolidation -= 1

            all_tasks.extend(tasks)
            remaining_total -= len(tasks)

        return all_tasks

    async def _claim_batch_for_schema(
        self, schema: str | None, limit: int, consolidation_limit: int
    ) -> list[ClaimedTask]:
        """
        Claim tasks from a specific schema respecting slot limits.

        实现原理：这是一个容错包装层。
        单个 schema claim 失败时，只记录日志并返回空结果，不让某个租户的
        表、索引或连接异常拖垮整个 worker 主轮询循环。
        """
        try:
            return await self._claim_batch_for_schema_inner(schema, limit, consolidation_limit)
        except Exception as e:
            # Format schema for logging: custom schemas in quotes, None as-is
            schema_display = f'"{schema}"' if schema else str(schema)
            logger.warning(f"Worker {self._worker_id} failed to claim tasks for schema {schema_display}: {e}")
            return []

    async def _claim_batch_for_schema_inner(
        self, schema: str | None, limit: int, consolidation_limit: int
    ) -> list[ClaimedTask]:
        """
        Inner implementation for claiming tasks from a specific schema with slot limits.

        实现原理：这里是整个 worker 领取任务的核心。
        它在一个数据库事务里完成三件事：
        1. 用 `FOR UPDATE SKIP LOCKED` 选出可领取的 pending 任务
        2. 把这些任务统一更新成 `processing`
        3. 把 `task_payload` 解析成内存里的 `ClaimedTask`

        把“选中”和“更新状态”放在同一个事务里，是为了保证分布式 worker
        并发下同一条任务不会被多个进程重复领取。
        """
        table = fq_table("async_operations", schema)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # 领取策略分两段：
                # 1. 先拿普通任务，尽量提升整体吞吐。
                # 2. 再拿 consolidation，但受独立限额控制。
                #
                # 这样做是因为 consolidation 往往更重，而且可能触发更高的 LLM
                # 或数据库成本；如果与普通任务共享同一配额，容易把 worker
                # 资源全部吃满，导致轻任务延迟变差。

                # 先 claim 非 consolidation 任务。
                # `FOR UPDATE SKIP LOCKED` 的关键含义是：
                # - `FOR UPDATE`：锁住当前选中的行
                # - `SKIP LOCKED`：如果别的 worker 已锁住某行，就跳过它
                #
                # 这样多个 worker 可以并发扫同一张表，但不会互相阻塞，也不会
                # 抢到同一条 pending 任务。
                non_consolidation_rows = await conn.fetch(
                    f"""
                    SELECT operation_id, task_payload, retry_count
                    FROM {table}
                    WHERE status = 'pending'
                      AND task_payload IS NOT NULL
                      AND operation_type != 'consolidation'
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                    ORDER BY created_at
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    limit,
                )

                claimed_count = len(non_consolidation_rows)
                remaining_limit = limit - claimed_count

                # consolidation 任务的 claim 附带了额外约束：
                # 同一个 bank 在同一时刻最多只允许一个 consolidation 处于
                # processing，避免同 bank 内部的聚合/刷新逻辑并发冲突。
                consolidation_rows = []
                if consolidation_limit > 0 and remaining_limit > 0:
                    consolidation_rows = await conn.fetch(
                        f"""
                        SELECT operation_id, task_payload, retry_count
                        FROM {table} AS pending
                        WHERE status = 'pending'
                          AND task_payload IS NOT NULL
                          AND operation_type = 'consolidation'
                          AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                          AND NOT EXISTS (
                              SELECT 1 FROM {table} AS processing
                              WHERE processing.bank_id = pending.bank_id
                                AND processing.operation_type = 'consolidation'
                                AND processing.status = 'processing'
                          )
                        ORDER BY created_at
                        LIMIT $1
                        FOR UPDATE SKIP LOCKED
                        """,
                        min(consolidation_limit, remaining_limit),
                    )

                all_rows = non_consolidation_rows + consolidation_rows

                if not all_rows:
                    return []

                # 只有在同一事务里把这些行改成 processing，claim 才算真正成立。
                # 如果只 SELECT 不 UPDATE，事务结束后别的 worker 仍可能再次选中
                # 这些任务，造成重复执行。
                operation_ids = [row["operation_id"] for row in all_rows]
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'processing', worker_id = $1, claimed_at = now(), updated_at = now()
                    WHERE operation_id = ANY($2)
                    """,
                    self._worker_id,
                    operation_ids,
                )

                # 任务在库里是 JSONB，worker 在内存里需要可执行的 dict。
                # 这里顺便注入 `_retry_count` 和 `_operation_id`，让执行阶段无需
                # 再回头查库就能知道重试次数和真实 operation 标识。
                result = []
                for row in all_rows:
                    task_dict = json.loads(row["task_payload"])
                    task_dict["_retry_count"] = row["retry_count"]
                    task_dict["_operation_id"] = str(row["operation_id"])
                    result.append(
                        ClaimedTask(
                            operation_id=str(row["operation_id"]),
                            task_dict=task_dict,
                            schema=schema,
                        )
                    )
                return result

    async def _mark_completed(self, operation_id: str, schema: str | None):
        """Mark a task as completed."""
        table = fq_table("async_operations", schema)
        await self._pool.execute(
            f"""
            UPDATE {table}
            SET status = 'completed', completed_at = now(), updated_at = now()
            WHERE operation_id = $1
            """,
            operation_id,
        )

    async def _mark_failed(self, operation_id: str, error_message: str, schema: str | None):
        """Mark a task as failed with error message, then propagate to parent if applicable."""
        table = fq_table("async_operations", schema)
        # Truncate error message if too long (max 5000 chars in schema)
        error_message = error_message[:5000] if len(error_message) > 5000 else error_message

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'failed', error_message = $2, completed_at = now(), updated_at = now()
                    WHERE operation_id = $1
                    """,
                    operation_id,
                    error_message,
                )
                await self._maybe_update_parent_operation(operation_id, schema, conn)

    async def _maybe_update_parent_operation(self, child_operation_id: str, schema: str | None, conn) -> None:
        """If this operation is a child of a batch_retain, update the parent status when all siblings are done.

        Must be called within an active transaction that has already updated the child's status.
        The memory engine has an equivalent method that runs inside task execution transactions.
        This poller-level version handles the case where a task fails via an unhandled exception
        that bypasses the memory engine's own failure path (e.g. a DB constraint violation that
        rolls back the engine's transaction before it can update the parent).
        """
        import json
        import uuid

        table = fq_table("async_operations", schema)

        try:
            row = await conn.fetchrow(
                f"SELECT result_metadata, bank_id FROM {table} WHERE operation_id = $1",
                uuid.UUID(child_operation_id),
            )
            if not row:
                return

            result_metadata = row["result_metadata"] or {}
            if isinstance(result_metadata, str):
                result_metadata = json.loads(result_metadata)
            parent_operation_id = result_metadata.get("parent_operation_id")
            if not parent_operation_id:
                return

            bank_id = row["bank_id"]

            # Lock parent to prevent concurrent sibling updates
            parent_row = await conn.fetchrow(
                f"SELECT operation_id FROM {table} WHERE operation_id = $1 AND bank_id = $2 FOR UPDATE",
                uuid.UUID(parent_operation_id),
                bank_id,
            )
            if not parent_row:
                return

            # Check whether all siblings are done
            siblings = await conn.fetch(
                f"""
                SELECT status FROM {table}
                WHERE bank_id = $1
                  AND result_metadata::jsonb @> $2::jsonb
                """,
                bank_id,
                json.dumps({"parent_operation_id": parent_operation_id}),
            )
            if not siblings or not all(s["status"] in ("completed", "failed") for s in siblings):
                return

            any_failed = any(s["status"] == "failed" for s in siblings)
            if any_failed:
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'failed', error_message = $2, updated_at = now()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(parent_operation_id),
                    "One or more sub-batches failed",
                )
            else:
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'completed', updated_at = now(), completed_at = now()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(parent_operation_id),
                )
            logger.info(
                f"Poller updated parent operation {parent_operation_id} to "
                f"{'failed' if any_failed else 'completed'} (all siblings done)"
            )
        except Exception as e:
            # Log but don't re-raise - the child has already been marked failed,
            # which is the critical state change. A stuck parent will be caught on
            # the next run or via monitoring.
            logger.error(f"Failed to update parent operation for child {child_operation_id}: {e}")

    async def _schedule_retry(self, operation_id: str, retry_at: "Any", error_message: str, schema: str | None):
        """Reset task to pending with a future retry timestamp."""
        table = fq_table("async_operations", schema)
        error_message = error_message[:5000] if len(error_message) > 5000 else error_message
        await self._pool.execute(
            f"""
            UPDATE {table}
            SET status = 'pending', next_retry_at = $2, worker_id = NULL, claimed_at = NULL,
                retry_count = retry_count + 1, error_message = $3, updated_at = now()
            WHERE operation_id = $1
            """,
            operation_id,
            retry_at,
            error_message,
        )
        logger.warning(f"Task {operation_id} scheduled for retry at {retry_at}: {error_message}")

    async def execute_task(self, task: ClaimedTask):
        """
        Execute a single task as a background job (fire-and-forget).

        实现原理：claim 和执行是解耦的。
        这里不会同步等待任务跑完，而是把每个已 claim 任务包装成后台
        `asyncio.Task`，这样 poller 主循环可以立刻回去继续 claim 下一批任务，
        从而把数据库领取和业务执行并行起来。
        """
        task_type = task.task_dict.get("type", "unknown")
        operation_type = task.task_dict.get("operation_type", "unknown")
        bank_id = task.task_dict.get("bank_id", "unknown")

        # fire-and-forget 让领取循环不被单个长任务阻塞；真正的生命周期管理
        # 依赖下面的本地 in-flight 计数和 done callback。
        bg_task = asyncio.create_task(self._execute_task_inner(task))

        # 本地计数是 slot 限流的真实依据，因此必须在任务启动后立即登记，
        # 否则主循环可能误以为还有空闲容量，继续超额 claim。
        async with self._in_flight_lock:
            self._active_tasks[task.operation_id] = (task_type, bank_id, task.schema, bg_task)
            self._in_flight_count += 1
            self._in_flight_by_type[operation_type] = self._in_flight_by_type.get(operation_type, 0) + 1

        # 用 done callback 统一回收本地状态，避免每个执行分支都手写清理逻辑。
        bg_task.add_done_callback(lambda _: asyncio.create_task(self._cleanup_task(task.operation_id, operation_type)))

    async def _cleanup_task(self, operation_id: str, operation_type: str):
        """Remove task from tracking after completion."""
        async with self._in_flight_lock:
            if operation_id in self._active_tasks:
                self._active_tasks.pop(operation_id, None)
                self._in_flight_count -= 1
                count = self._in_flight_by_type.get(operation_type, 0)
                if count > 0:
                    self._in_flight_by_type[operation_type] = count - 1
                    if self._in_flight_by_type[operation_type] == 0:
                        del self._in_flight_by_type[operation_type]

    async def _execute_task_inner(self, task: ClaimedTask):
        """Inner task execution with retry/fail handling.

        Tasks that want to be retried raise RetryTaskAt; the poller sets next_retry_at
        and resets status to 'pending'. All other exceptions are marked as failed immediately.
        Non-retryable failures (e.g., file_convert_retain) are handled by the executor
        internally - it marks the operation as failed and returns normally.

        实现原理：poller 只负责通用调度语义，不负责理解具体业务任务。
        所以这里把执行结果压缩成三类：
        - 正常返回：任务完成
        - 抛 `RetryTaskAt`：任务希望延后重试
        - 抛其他异常：任务失败，poller 统一落失败状态

        这种分层让具体 handler 只表达“业务是否需要重试”，而不需要每个 handler
        自己重复写一遍状态机更新逻辑。
        """
        task_type = task.task_dict.get("type", "unknown")
        bank_id = task.task_dict.get("bank_id", "unknown")

        try:
            schema_info = f", schema={task.schema}" if task.schema else ""
            logger.debug(f"Executing task {task.operation_id} (type={task_type}, bank={bank_id}{schema_info})")
            if task.schema:
                # schema 不是业务 payload 的一部分，而是执行上下文的一部分。
                # 这里在真正调用 executor 前注入 `_schema`，让 MemoryEngine 能在
                # 正确的租户 schema 下查询和更新数据。
                task.task_dict["_schema"] = task.schema
            await self._executor(task.task_dict)
            logger.debug(f"Task {task.operation_id} execution finished")
        except RetryTaskAt as e:
            await self._schedule_retry(task.operation_id, e.retry_at, str(e), task.schema)
        except Exception as e:
            logger.error(f"Task {task.operation_id} failed: {e}")
            traceback.print_exc()
            await self._mark_failed(task.operation_id, str(e), task.schema)

    async def recover_own_tasks(self) -> int:
        """
        Recover tasks that were assigned to this worker but not completed.

        This handles the case where a worker crashes while processing tasks.
        On startup, we reset any tasks stuck in 'processing' for this worker_id
        back to 'pending' so they can be picked up again.

        Also recovers batch API operations that were in-flight.

        If tenant_extension is configured, recovers across all tenant schemas.

        Returns:
            Number of tasks recovered

        实现原理：worker 崩溃时，数据库里可能残留一批 `processing` 任务。
        如果不在重启时回收，这些任务会永远卡住，因为普通 claim 逻辑只看
        `pending` 状态。

        因此 worker 启动后会先把“属于自己 worker_id 的 processing 任务”
        重置回 pending，让它们重新进入可领取队列。
        """
        schemas = await self._get_schemas()
        total_count = 0

        for schema in schemas:
            try:
                table = fq_table("async_operations", schema)

                # batch API 任务有额外生命周期，先单独恢复，再恢复普通任务，
                # 避免两套恢复规则互相覆盖。
                batch_count = await self._recover_batch_operations(schema)
                total_count += batch_count

                # 这里只恢复“上次由本 worker 认领”的 processing 任务，
                # 不碰其他 worker 的进行中任务，避免误抢活跃实例的工作。
                result = await self._pool.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
                    WHERE status = 'processing' AND worker_id = $1 AND result_metadata->>'batch_id' IS NULL
                    """,
                    self._worker_id,
                )

                # Parse "UPDATE N" to get count
                count = int(result.split()[-1]) if result else 0
                total_count += count
            except Exception as e:
                # Format schema for logging: custom schemas in quotes, None as-is
                schema_display = f'"{schema}"' if schema else str(schema)
                logger.warning(f"Worker {self._worker_id} failed to recover tasks for schema {schema_display}: {e}")

        if total_count > 0:
            logger.info(f"Worker {self._worker_id} recovered {total_count} stale tasks from previous run")
        return total_count

    async def _recover_batch_operations(self, schema: str | None) -> int:
        """
        Recover batch API operations that were in-flight when worker crashed.

        Finds operations with batch_id in metadata and re-submits them as tasks
        so polling can resume.

        Args:
            schema: Database schema to recover from

        Returns:
            Number of batch operations recovered
        """
        table = fq_table("async_operations", schema)

        try:
            # Find operations with batch_id in metadata (batch API operations)
            rows = await self._pool.fetch(
                f"""
                SELECT operation_id, task_payload, result_metadata
                FROM {table}
                WHERE status = 'processing'
                  AND result_metadata ? 'batch_id'
                  AND task_payload IS NOT NULL
                """
            )

            if not rows:
                return 0

            recovered = 0
            for row in rows:
                operation_id = str(row["operation_id"])
                task_payload = row["task_payload"]
                result_metadata = row["result_metadata"]

                # Parse metadata
                if isinstance(result_metadata, str):
                    result_metadata = json.loads(result_metadata)

                batch_id = result_metadata.get("batch_id")
                batch_provider = result_metadata.get("batch_provider", "openai")

                logger.info(
                    f"Recovering batch operation: operation_id={operation_id}, batch_id={batch_id}, provider={batch_provider}"
                )

                # Parse task_payload
                if isinstance(task_payload, str):
                    task_dict = json.loads(task_payload)
                else:
                    task_dict = task_payload

                # Mark operation as ready for re-processing
                # Reset to pending with task_payload intact so worker picks it up again
                await self._pool.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
                    WHERE operation_id = $1
                    """,
                    operation_id,
                )

                recovered += 1
                logger.info(f"Batch operation {operation_id} reset to pending for re-processing")

            return recovered

        except Exception as e:
            schema_display = f'"{schema}"' if schema else str(schema)
            logger.error(f"Failed to recover batch operations for schema {schema_display}: {e}")
            return 0

    async def run(self):
        """
        Main polling loop with fire-and-forget task execution.

        Continuously polls for pending tasks, spawns them as background tasks,
        and immediately continues polling (up to slot limits).

        实现原理：这个循环本质上是一个轻量调度器：
        - 启动先恢复上次异常退出遗留的 processing 任务
        - 每轮 claim 一批任务
        - 把它们交给后台执行
        - 如果本轮领到了任务，就立刻继续下一轮，尽快填满可用 slot
        - 如果没领到任务，再按 poll_interval 休眠

        这种“有活就快速连拉、没活才 sleep”的模式，比固定周期只拉一批
        吞吐更高，也能在队列变空时避免忙等。
        """
        await self.recover_own_tasks()

        logger.info(
            f"Worker {self._worker_id} starting polling loop "
            f"(max_slots={self._max_slots}, consolidation_max_slots={self._consolidation_max_slots})"
        )

        while not self._shutdown.is_set():
            try:
                # claim 是同步拿任务，执行是异步后台跑；两者分离后，主循环就能
                # 尽量持续把本 worker 的并发槽位填满。
                tasks = await self.claim_batch()

                if tasks:
                    # Log batch info
                    task_types: dict[str, int] = {}
                    schemas_seen: set[str | None] = set()
                    consolidation_count = 0
                    for task in tasks:
                        t = task.task_dict.get("type", "unknown")
                        op_type = task.task_dict.get("operation_type", "unknown")
                        task_types[t] = task_types.get(t, 0) + 1
                        schemas_seen.add(task.schema)
                        if op_type == "consolidation":
                            consolidation_count += 1

                    types_str = ", ".join(f"{k}:{v}" for k, v in task_types.items())
                    # Display None as "default" in logs
                    schemas_str = ", ".join(s if s else "default" for s in schemas_seen)
                    logger.info(
                        f"Worker {self._worker_id} claimed {len(tasks)} tasks "
                        f"({consolidation_count} consolidation): {types_str} (schemas: {schemas_str})"
                    )

                    # 这里只负责把任务发射出去，不等待完成；slot 回收由
                    # execute_task -> done callback -> _cleanup_task 负责。
                    for task in tasks:
                        await self.execute_task(task)

                    # 本轮既然拿到了任务，就说明队列里大概率还有货，立即进入
                    # 下一轮比 sleep 更能提高吞吐。
                    continue

                # 没有任务可领时再 sleep，避免对数据库形成空转轮询压力。
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(),
                        timeout=self._poll_interval_ms / 1000,
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue polling

                # Log progress stats periodically
                await self._log_progress_if_due()

            except asyncio.CancelledError:
                logger.info(f"Worker {self._worker_id} polling loop cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {self._worker_id} error in polling loop: {e}")
                traceback.print_exc()
                # Backoff on error
                await asyncio.sleep(1)

        logger.info(f"Worker {self._worker_id} polling loop stopped")

    async def shutdown_graceful(self, timeout: float = 30.0):
        """
        Signal shutdown and wait for current tasks to complete.

        Args:
            timeout: Maximum time to wait for in-flight tasks (seconds)
        """
        logger.info(f"Worker {self._worker_id} initiating graceful shutdown")
        self._shutdown.set()

        # Wait for in-flight tasks to complete
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            async with self._in_flight_lock:
                in_flight = self._in_flight_count
                active_task_objects = [task_info[3] for task_info in self._active_tasks.values()]

            if in_flight == 0:
                logger.info(f"Worker {self._worker_id} graceful shutdown complete")
                return

            logger.info(f"Worker {self._worker_id} waiting for {in_flight} in-flight tasks")

            # Wait for at least one task to complete
            if active_task_objects:
                done, _ = await asyncio.wait(active_task_objects, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(0.5)

        logger.warning(f"Worker {self._worker_id} shutdown timeout after {timeout}s, cancelling remaining tasks")

        # Cancel remaining tasks
        async with self._in_flight_lock:
            for operation_id, (_, _, _, bg_task) in list(self._active_tasks.items()):
                if not bg_task.done():
                    bg_task.cancel()

    async def _log_progress_if_due(self):
        """Log progress stats every PROGRESS_LOG_INTERVAL seconds."""
        now = time.time()
        if now - self._last_progress_log < PROGRESS_LOG_INTERVAL:
            return

        self._last_progress_log = now

        try:
            # Get local active tasks
            async with self._in_flight_lock:
                in_flight = self._in_flight_count
                in_flight_by_type = dict(self._in_flight_by_type)
                active_tasks = dict(self._active_tasks)

            consolidation_count = in_flight_by_type.get("consolidation", 0)
            available_slots = self._max_slots - in_flight
            available_consolidation_slots = self._consolidation_max_slots - consolidation_count

            # Build local processing breakdown
            task_groups: dict[tuple[str, str], int] = {}
            for op_type, bank_id, _, _ in active_tasks.values():
                key = (op_type, bank_id)
                task_groups[key] = task_groups.get(key, 0) + 1

            processing_info = [f"{op}:{bank}({cnt})" for (op, bank), cnt in task_groups.items()]
            processing_str = ", ".join(processing_info[:10]) if processing_info else "none"
            if len(processing_info) > 10:
                processing_str += f" +{len(processing_info) - 10} more"

            # Get global stats from DB
            schemas = await self._get_schemas()
            global_pending = 0
            all_worker_counts: dict[str, int] = {}

            async with self._pool.acquire() as conn:
                for schema in schemas:
                    table = fq_table("async_operations", schema)

                    row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {table} WHERE status = 'pending'")
                    global_pending += row["count"] if row else 0

                    worker_rows = await conn.fetch(
                        f"""
                        SELECT worker_id, COUNT(*) as count
                        FROM {table}
                        WHERE status = 'processing'
                        GROUP BY worker_id
                        """
                    )
                    for wr in worker_rows:
                        wid = wr["worker_id"] or "unknown"
                        all_worker_counts[wid] = all_worker_counts.get(wid, 0) + wr["count"]

            other_workers = []
            for wid, cnt in all_worker_counts.items():
                if wid != self._worker_id:
                    other_workers.append(f"{wid}:{cnt}")
            others_str = ", ".join(other_workers) if other_workers else "none"

            # Display None as "default" in logs
            schemas_str = ", ".join(s if s else "default" for s in schemas)
            logger.info(
                f"[WORKER_STATS] worker={self._worker_id} "
                f"slots={in_flight}/{self._max_slots} (consolidation={consolidation_count}/{self._consolidation_max_slots}) | "
                f"available={available_slots} (consolidation={available_consolidation_slots}) | "
                f"global: pending={global_pending} (schemas: {schemas_str}) | "
                f"others: {others_str} | "
                f"my_active: {processing_str}"
            )

        except Exception as e:
            logger.debug(f"Failed to log progress stats: {e}")

    @property
    def worker_id(self) -> str:
        """Get the worker ID."""
        return self._worker_id

    @property
    def is_shutdown(self) -> bool:
        """Check if shutdown has been signaled."""
        return self._shutdown.is_set()
