from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable


JSONRPC_VERSION = "2.0"


class CodexAppServerError(RuntimeError):
    """Raised when Codex app-server transport/protocol handling fails."""


@dataclass(frozen=True, slots=True)
class CodexTurnResult:
    thread_id: str
    turn_id: str | None
    status: str
    usage: dict[str, Any] | None
    thread_usage: dict[str, Any] | None
    start_result: dict[str, Any] | None
    completion: dict[str, Any]
    notifications: list[dict[str, Any]]
    server_requests: list[dict[str, Any]]


class CodexAppServerSession:
    """Minimal JSON-RPC client for `codex app-server --listen stdio://`."""

    def __init__(
        self,
        *,
        codex_path: str = "codex",
        app_args: Iterable[str] | None = None,
        timeout_s: int = 300,
    ) -> None:
        self.codex_path = str(codex_path)
        self.app_args = [str(x) for x in (app_args or [])]
        self.timeout_s = int(timeout_s)

        self._proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._pending_responses: dict[int | str, dict[str, Any]] = {}
        self._request_id = 1
        self._initialized = False
        self._lock = threading.Lock()

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        self._initialized = False
        with self._lock:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def ensure_initialized(self) -> None:
        self._ensure_started()
        if self._initialized:
            return
        params = {
            "clientInfo": {"name": "games-bench", "version": "0.1.0"},
            "capabilities": {"experimentalApi": True},
        }
        self.request("initialize", params=params, timeout_s=self.timeout_s)
        self._send({"jsonrpc": JSONRPC_VERSION, "method": "initialized", "params": {}})
        self._initialized = True

    def request(
        self,
        method: str,
        *,
        params: dict[str, Any] | None = None,
        timeout_s: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_started()
        request_id = self._next_request_id()
        payload: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        self._send(payload)
        return self._wait_for_response(request_id, timeout_s=timeout_s)

    def run_turn(
        self,
        *,
        model: str | None,
        input_text: str,
        dynamic_tools: list[dict[str, Any]],
        on_tool_call: Callable[[dict[str, Any]], dict[str, Any]],
        output_schema: dict[str, Any] | None = None,
        timeout_s: int | None = None,
    ) -> CodexTurnResult:
        self.ensure_initialized()
        turn_timeout = int(timeout_s or self.timeout_s)
        deadline = time.monotonic() + max(1, turn_timeout)

        thread_params: dict[str, Any] = {
            "ephemeral": True,
            "experimentalRawEvents": False,
            "persistExtendedHistory": False,
            "dynamicTools": dynamic_tools,
        }
        if model:
            thread_params["model"] = model
        thread_result = self.request(
            "thread/start",
            params=thread_params,
            timeout_s=turn_timeout,
        )
        thread_id = str(thread_result.get("threadId") or "").strip()
        if not thread_id:
            thread_obj = thread_result.get("thread")
            if isinstance(thread_obj, dict):
                thread_id = str(thread_obj.get("id") or "").strip()
        if not thread_id:
            raise CodexAppServerError("thread/start did not return threadId.")

        turn_params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": input_text}],
        }
        if output_schema is not None:
            turn_params["outputSchema"] = output_schema

        turn_request_id = self._next_request_id()
        self._send(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": turn_request_id,
                "method": "turn/start",
                "params": turn_params,
            }
        )

        turn_start_result: dict[str, Any] | None = None
        turn_id: str | None = None
        notifications: list[dict[str, Any]] = []
        server_requests: list[dict[str, Any]] = []
        completion_params: dict[str, Any] | None = None
        thread_usage: dict[str, Any] | None = None

        while completion_params is None:
            msg = self._recv_message(deadline=deadline)

            if "id" in msg and "method" not in msg:
                response_id = msg.get("id")
                if response_id == turn_request_id:
                    if "error" in msg:
                        raise CodexAppServerError(
                            f"turn/start failed: {json.dumps(msg['error'], sort_keys=True)}"
                        )
                    result = msg.get("result")
                    if isinstance(result, dict):
                        turn_start_result = result
                        started_turn_id = result.get("turnId")
                        if not started_turn_id and isinstance(result.get("turn"), dict):
                            started_turn_id = result["turn"].get("id")
                        if isinstance(started_turn_id, str) and started_turn_id:
                            turn_id = started_turn_id
                    continue
                self._pending_responses[response_id] = msg
                continue

            if "id" in msg and "method" in msg:
                request_id = msg.get("id")
                method = str(msg.get("method") or "")
                params = msg.get("params")
                if not isinstance(params, dict):
                    params = {}
                server_requests.append({"method": method, "params": params})
                if method == "item/tool/call":
                    try:
                        tool_result = on_tool_call(params)
                        if not isinstance(tool_result, dict):
                            tool_result = _tool_result(
                                False, "Tool handler returned non-object response."
                            )
                    except Exception as exc:  # pragma: no cover
                        tool_result = _tool_result(False, f"Tool handler error: {exc}")
                    self._send(
                        {
                            "jsonrpc": JSONRPC_VERSION,
                            "id": request_id,
                            "result": tool_result,
                        }
                    )
                else:
                    self._send(
                        {
                            "jsonrpc": JSONRPC_VERSION,
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": f"Unsupported server request: {method}",
                            },
                        }
                    )
                continue

            method = str(msg.get("method") or "")
            params = msg.get("params")
            if not isinstance(params, dict):
                params = {}
            notifications.append({"method": method, "params": params})

            if method == "thread/tokenUsage/updated":
                usage = params.get("tokenUsage")
                if isinstance(usage, dict):
                    thread_usage = usage
                continue

            if method != "turn/completed":
                continue
            completed_thread = str(params.get("threadId") or "")
            if completed_thread and completed_thread != thread_id:
                continue
            completion_params = params
            completed_turn_id = params.get("turnId")
            if not completed_turn_id and isinstance(params.get("turn"), dict):
                completed_turn_id = params["turn"].get("id")
            if isinstance(completed_turn_id, str) and completed_turn_id:
                turn_id = completed_turn_id

        status = str(completion_params.get("status") or "")
        if not status and isinstance(completion_params.get("turn"), dict):
            status = str(completion_params["turn"].get("status") or "")
        status = status or "unknown"
        usage = completion_params.get("usage")
        if not usage and isinstance(completion_params.get("turn"), dict):
            usage = completion_params["turn"].get("usage")
        normalized_usage = usage if isinstance(usage, dict) else None
        return CodexTurnResult(
            thread_id=thread_id,
            turn_id=turn_id,
            status=status,
            usage=normalized_usage,
            thread_usage=thread_usage,
            start_result=turn_start_result,
            completion=completion_params,
            notifications=notifications,
            server_requests=server_requests,
        )

    def _ensure_started(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            cmd = [
                self.codex_path,
                "app-server",
                "--listen",
                "stdio://",
                *self.app_args,
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                raise CodexAppServerError(
                    f"Failed to start codex app-server: {exc}"
                ) from exc

            stdout = self._proc.stdout
            stderr = self._proc.stderr
            if stdout is None or stderr is None or self._proc.stdin is None:
                raise CodexAppServerError(
                    "Failed to open stdio pipes for codex app-server."
                )

            threading.Thread(
                target=self._stdout_reader,
                args=(stdout,),
                daemon=True,
            ).start()
            threading.Thread(
                target=self._stderr_reader,
                args=(stderr,),
                daemon=True,
            ).start()

    def _stdout_reader(self, stream: Any) -> None:
        try:
            for raw_line in stream:
                line = str(raw_line).strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self._stderr_tail.append(f"[stdout-non-json] {line}")
                    continue
                if isinstance(msg, dict):
                    self._stdout_queue.put(msg)
        finally:
            self._stdout_queue.put({"__eof__": True})

    def _stderr_reader(self, stream: Any) -> None:
        try:
            for raw_line in stream:
                line = str(raw_line).rstrip()
                if line:
                    self._stderr_tail.append(line)
        finally:
            pass

    def _next_request_id(self) -> int:
        value = self._request_id
        self._request_id += 1
        return value

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise CodexAppServerError("codex app-server process is not running.")
        if proc.poll() is not None:
            raise CodexAppServerError(
                "codex app-server process exited unexpectedly."
                + self._format_stderr_tail()
            )
        wire = json.dumps(payload, ensure_ascii=True)
        try:
            proc.stdin.write(wire + "\n")
            proc.stdin.flush()
        except OSError as exc:
            raise CodexAppServerError(
                f"Failed to write to codex app-server: {exc}"
            ) from exc

    def _wait_for_response(
        self,
        request_id: int,
        *,
        timeout_s: int | None,
    ) -> dict[str, Any]:
        pending = self._pending_responses.pop(request_id, None)
        if pending is not None:
            return self._parse_response(pending, request_id=request_id)

        timeout = max(1, int(timeout_s or self.timeout_s))
        deadline = time.monotonic() + timeout
        while True:
            msg = self._recv_message(deadline=deadline)
            if "id" in msg and "method" not in msg:
                response_id = msg.get("id")
                if response_id == request_id:
                    return self._parse_response(msg, request_id=request_id)
                self._pending_responses[response_id] = msg
                continue
            if "id" in msg and "method" in msg:
                server_request_id = msg.get("id")
                method = str(msg.get("method") or "")
                self._send(
                    {
                        "jsonrpc": JSONRPC_VERSION,
                        "id": server_request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Unsupported server request while waiting for response: {method}",
                        },
                    }
                )
                continue

    def _parse_response(
        self,
        msg: dict[str, Any],
        *,
        request_id: int,
    ) -> dict[str, Any]:
        if "error" in msg:
            raise CodexAppServerError(
                f"Request id {request_id} failed: {json.dumps(msg['error'], sort_keys=True)}"
            )
        result = msg.get("result")
        if isinstance(result, dict):
            return result
        if result is None:
            return {}
        raise CodexAppServerError(
            f"Request id {request_id} returned non-object result: {result!r}"
        )

    def _recv_message(self, *, deadline: float) -> dict[str, Any]:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise CodexAppServerError(
                "Timed out waiting for codex app-server message."
                + self._format_stderr_tail()
            )
        try:
            msg = self._stdout_queue.get(timeout=timeout)
        except queue.Empty:
            raise CodexAppServerError(
                "Timed out waiting for codex app-server output."
                + self._format_stderr_tail()
            )
        if msg.get("__eof__"):
            raise CodexAppServerError(
                "codex app-server closed stdout unexpectedly."
                + self._format_stderr_tail()
            )
        return msg

    def _format_stderr_tail(self) -> str:
        if not self._stderr_tail:
            return ""
        lines = list(self._stderr_tail)[-20:]
        return "\nRecent codex app-server stderr:\n" + "\n".join(lines)

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


def _tool_result(success: bool, text: str) -> dict[str, Any]:
    return {
        "success": bool(success),
        "contentItems": [{"type": "inputText", "text": str(text)}],
    }
