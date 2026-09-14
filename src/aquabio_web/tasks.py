"""Bounded background jobs with cooperative cancellation and progress events."""
from concurrent.futures import ThreadPoolExecutor, CancelledError
from datetime import datetime, timezone
import threading
import time
import uuid

from aquabio_mrag.cancellation import cancel_event, check_cancelled


class TaskQueueFull(ValueError):
    pass


class BackgroundTasks:
    def _init_tasks(self, max_workers=4, max_pending=16, retention_seconds=3600, max_completed=100):
        self._tasks = {}
        self._task_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aquabio")
        self._task_slots = threading.BoundedSemaphore(max_pending)
        self._retention_seconds = retention_seconds
        self._max_completed = max_completed
        self._closed = False

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    def _prune_tasks(self):
        now = time.monotonic()
        finished = sorted(
            ((key, task) for key, task in self._tasks.items() if task.get("finished_monotonic")),
            key=lambda item: item[1]["finished_monotonic"], reverse=True,
        )
        for index, (key, task) in enumerate(finished):
            if index >= self._max_completed or now - task["finished_monotonic"] >= self._retention_seconds:
                del self._tasks[key]

    def submit_chat(self, request):
        with self._task_lock:
            if self._closed:
                raise TaskQueueFull("聊天服务正在关闭")
            self._prune_tasks()
            if not self._task_slots.acquire(blocking=False):
                raise TaskQueueFull("聊天任务队列已满，请稍后重试")
            task_id = f"task_{uuid.uuid4().hex[:16]}"
            self._tasks[task_id] = {
                "task_id": task_id, "session_id": request.session_id,
                "status": "queued", "stage": "排队等待", "detail": "请求已进入后台任务队列。",
                "created_at": self._utc_now(), "started_at": "", "finished_at": "",
                "elapsed_seconds": 0, "result": None, "error": "",
                "cancel_requested": False, "progress": [], "revision": 0,
                "cancel_event": threading.Event(),
            }
            try:
                self._executor.submit(self._run_chat_task, task_id, request)
            except BaseException:
                del self._tasks[task_id]
                self._task_slots.release()
                raise
            return self.chat_task(task_id)

    def _set_task(self, task_id, **values):
        with self._task_lock:
            self._tasks[task_id].update(values)

    def _run_chat_task(self, task_id, request):
        started = time.monotonic()
        with self._task_lock:
            event = self._tasks[task_id]["cancel_event"]
        token = cancel_event.set(event)
        try:
            check_cancelled()
            self._set_task(task_id, status="running", stage="执行 LangGraph", started_at=self._utc_now())
            def report(value):
                check_cancelled()
                node, _, detail = value.partition(":")
                with self._task_lock:
                    task = self._tasks[task_id]
                    task["revision"] += 1
                    task["progress"].append({"id": task["revision"], "node": node, "detail": detail})
                    task["progress"] = task["progress"][-100:]
                    task.update(stage=f"LangGraph: {node}", detail=detail or value)
            result = self.chat(request, progress_callback=report)
            check_cancelled()
            self._set_task(task_id, status="completed", stage="完成", detail="答案与证据已生成。", result=result)
        except CancelledError:
            self._set_task(task_id, status="cancelled", stage="已取消", detail="已停止后续处理。", result=None)
        except Exception as error:
            self._set_task(task_id, status="failed", stage="执行失败", error=f"{type(error).__name__}: {error}")
        finally:
            cancel_event.reset(token)
            self._set_task(task_id, finished_at=self._utc_now(), finished_monotonic=time.monotonic(),
                           elapsed_seconds=round(time.monotonic()-started, 1))
            self._task_slots.release()

    def chat_task(self, task_id):
        with self._task_lock:
            task = {k: v for k, v in self._tasks[task_id].items()
                    if k not in {"cancel_event", "finished_monotonic"}}
            task["progress"] = list(task["progress"])
        if task["status"] in {"queued", "running"} and task["started_at"]:
            task["elapsed_seconds"] = round((datetime.now(timezone.utc)-datetime.fromisoformat(task["started_at"])).total_seconds(), 1)
        return task

    def chat_tasks(self):
        with self._task_lock:
            self._prune_tasks()
            return [self.chat_task(key) for key in reversed(self._tasks)]

    def cancel_chat_task(self, task_id):
        with self._task_lock:
            task = self._tasks[task_id]
            if task["status"] in {"queued", "running"}:
                task["cancel_requested"] = True
                task["cancel_event"].set()
                task["stage"] = "正在取消"
                task["detail"] = "当前调用结束后停止后续节点。"
            return self.chat_task(task_id)

    def close(self):
        with self._task_lock:
            self._closed = True
            for task in self._tasks.values():
                if task["status"] in {"queued", "running"}:
                    task["cancel_event"].set()
        self._executor.shutdown(wait=True)
        for workflow in getattr(self, "_workflows", {}).values():
            connection = getattr(workflow, "_checkpoint_connection", None)
            if connection is not None:
                connection.close()
