from __future__ import annotations

import json
import os
import sys
import shlex
import subprocess
import time
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProviderResult:
    tool_calls: list[ToolCall]
    raw: Any
    error: str | None = None
    usage: dict[str, Any] | None = None
    cost: float | None = None


def _tool_call_from_name_and_arguments(
    name: Any,
    arguments: Any,
) -> ToolCall | None:
    if name is None:
        return None
    parsed_arguments = arguments
    if isinstance(parsed_arguments, str):
        parsed = _parse_json_loose(parsed_arguments)
        if isinstance(parsed, dict):
            parsed_arguments = parsed
    if not isinstance(parsed_arguments, dict):
        parsed_arguments = {"value": parsed_arguments}
    return ToolCall(name=str(name), arguments=parsed_arguments)


def _normalize_tool_call_mapping(obj: dict[str, Any]) -> list[ToolCall]:
    if "tool_call" in obj:
        return _normalize_tool_calls(obj["tool_call"])
    if "tool_calls" in obj:
        return _normalize_tool_calls(obj["tool_calls"])
    if "name" in obj and "arguments" in obj:
        call = _tool_call_from_name_and_arguments(obj.get("name"), obj.get("arguments"))
        return [call] if call is not None else []
    if "function" in obj and isinstance(obj["function"], dict):
        func = obj["function"]
        if "name" in func and "arguments" in func:
            call = _tool_call_from_name_and_arguments(
                func.get("name"),
                func.get("arguments"),
            )
            return [call] if call is not None else []
    for key in ("content", "message", "output", "text"):
        if key in obj:
            normalized = _normalize_tool_calls(obj[key])
            if normalized:
                return normalized
    return []


def _normalize_tool_calls(obj: Any) -> list[ToolCall]:
    if obj is None:
        return []
    if isinstance(obj, ToolCall):
        return [obj]
    if isinstance(obj, str):
        parsed = _parse_json_loose(obj)
        if parsed is not None:
            return _normalize_tool_calls(parsed)
        return []
    if isinstance(obj, list):
        calls: list[ToolCall] = []
        for item in obj:
            calls.extend(_normalize_tool_calls(item))
        return calls
    if isinstance(obj, dict):
        return _normalize_tool_call_mapping(obj)

    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except Exception:
            dumped = None
        if dumped is not None and dumped is not obj:
            normalized = _normalize_tool_calls(dumped)
            if normalized:
                return normalized

    obj_type = getattr(obj, "type", None)
    if obj_type == "function_call":
        call = _tool_call_from_name_and_arguments(
            getattr(obj, "name", None),
            getattr(obj, "arguments", None),
        )
        if call is not None:
            return [call]

    for key in ("tool_call", "tool_calls", "content", "message", "output", "text"):
        if hasattr(obj, key):
            normalized = _normalize_tool_calls(getattr(obj, key))
            if normalized:
                return normalized

    function_obj = getattr(obj, "function", None)
    if function_obj is not None:
        call = _tool_call_from_name_and_arguments(
            getattr(function_obj, "name", None),
            getattr(function_obj, "arguments", None),
        )
        if call is not None:
            return [call]

    call = _tool_call_from_name_and_arguments(
        getattr(obj, "name", None),
        getattr(obj, "arguments", None),
    )
    if call is not None and (
        obj_type in {"function", "tool_call", "function_call"}
        or hasattr(obj, "arguments")
    ):
        return [call]
    return []


def _parse_json_loose(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


def _normalize_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        return dump()
    if hasattr(usage, "__dict__"):
        return dict(usage.__dict__)
    return None


def _extract_cost(
    usage: dict[str, Any] | None, raw: dict[str, Any] | None
) -> float | None:
    if usage:
        for key in ("total_cost", "cost", "total_cost_usd", "total_price"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    if raw:
        for key in ("cost", "total_cost", "total_cost_usd", "total_price"):
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _tool_schemas_to_openai_chat(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        }
        for t in tools
    ]


def _tool_schemas_to_openai_responses(
    tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {}),
        }
        for t in tools
    ]


def _missing_openai_dependency_message(exc: Exception) -> str:
    return (
        "Missing dependency: openai SDK. Install with "
        "pip install 'games-bench[llm]' or uv sync --group llm. "
        f"ImportError: {exc}"
    )


class OpenRouterProvider:
    """OpenRouter provider using OpenAI-compatible Chat Completions."""

    supports_images = True

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        stream_debug: bool = False,
        timeout_s: int = 300,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY")
        self.http_referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        self.x_title = x_title or os.environ.get("OPENROUTER_X_TITLE")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.stream_debug = bool(stream_debug)
        self.timeout_s = int(timeout_s)

    def _stream_log(self, message: str) -> None:
        if not self.stream_debug:
            return
        print(f"[openrouter stream-debug] {message}", file=sys.stderr, flush=True)

    def _completion_payload(
        self, client: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.stream_debug:
            return self._rest_completion_payload(kwargs)
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("timeout", float(self.timeout_s))
        return self._stream_completion_payload(client, call_kwargs)

    def _rest_completion_payload(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            data=json.dumps(kwargs).encode("utf-8"),
            headers=self._rest_headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.timeout_s)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Request failed: {exc}") from exc
        if isinstance(payload, dict):
            return payload
        raise RuntimeError("Provider error: non-dict JSON payload from OpenRouter")

    def _rest_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title
        return headers

    def _stream_completion_payload(
        self, client: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream_kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = client.chat.completions.create(**stream_kwargs)
        except Exception:
            stream_kwargs.pop("stream_options", None)
            stream = client.chat.completions.create(**stream_kwargs)

        started = time.monotonic()
        last_tick = started
        chunk_count = 0
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        choice_error: dict[str, Any] | None = None

        for chunk in stream:
            chunk_count += 1
            now = time.monotonic()
            elapsed_s = now - started
            delta_s = now - last_tick
            last_tick = now

            payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            if not isinstance(payload, dict):
                self._stream_log(
                    f"+{elapsed_s:.2f}s dt={delta_s:.2f}s chunk={chunk_count:03d} non-dict payload"
                )
                continue

            usage_chunk = _normalize_usage(payload.get("usage"))
            if usage_chunk:
                usage = usage_chunk

            choices = payload.get("choices") or []
            if not choices:
                self._stream_log(
                    f"+{elapsed_s:.2f}s dt={delta_s:.2f}s chunk={chunk_count:03d} no choices"
                )
                continue

            choice = choices[0]
            if not isinstance(choice, dict):
                self._stream_log(
                    f"+{elapsed_s:.2f}s dt={delta_s:.2f}s chunk={chunk_count:03d} non-dict choice"
                )
                continue
            if isinstance(choice.get("error"), dict):
                choice_error = choice["error"]

            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish_reason = choice.get("finish_reason")

            parts: list[str] = []
            content_piece = delta.get("content")
            if isinstance(content_piece, str) and content_piece:
                content_parts.append(content_piece)
                parts.append(f"content+={len(content_piece)}")

            tool_call_deltas = delta.get("tool_calls") or []
            if isinstance(tool_call_deltas, list):
                for tool_call_delta in tool_call_deltas:
                    if not isinstance(tool_call_delta, dict):
                        continue
                    raw_index = tool_call_delta.get("index", 0)
                    try:
                        index = int(raw_index)
                    except (TypeError, ValueError):
                        index = 0
                    entry = tool_calls_by_index.setdefault(
                        index,
                        {
                            "type": "function",
                            "id": None,
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    tool_id = tool_call_delta.get("id")
                    if isinstance(tool_id, str) and tool_id:
                        entry["id"] = tool_id

                    function_delta = tool_call_delta.get("function") or {}
                    if isinstance(function_delta, dict):
                        name = function_delta.get("name")
                        if isinstance(name, str) and name:
                            entry["function"]["name"] = name
                        arguments_chunk = function_delta.get("arguments")
                        if isinstance(arguments_chunk, str) and arguments_chunk:
                            entry["function"]["arguments"] += arguments_chunk
                            parts.append(
                                f"tool[{index}]={entry['function']['name'] or '?'} args+={len(arguments_chunk)}"
                            )
                        elif isinstance(name, str) and name:
                            parts.append(f"tool[{index}]={name}")

            if not parts:
                parts.append("delta metadata only")
            self._stream_log(
                f"+{elapsed_s:.2f}s dt={delta_s:.2f}s chunk={chunk_count:03d} {'; '.join(parts)}"
            )

        tool_calls: list[dict[str, Any]] = []
        for index in sorted(tool_calls_by_index):
            entry = tool_calls_by_index[index]
            function_data = entry.get("function") or {}
            tool_call: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": function_data.get("name", ""),
                    "arguments": function_data.get("arguments", ""),
                },
            }
            tool_id = entry.get("id")
            if isinstance(tool_id, str) and tool_id:
                tool_call["id"] = tool_id
            tool_calls.append(tool_call)

        message: dict[str, Any] = {}
        if content_parts:
            message["content"] = "".join(content_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls

        data: dict[str, Any] = {
            "choices": [
                {
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ]
        }
        if usage:
            data["usage"] = usage
        if choice_error is not None:
            data["choices"][0]["error"] = choice_error

        total_elapsed = time.monotonic() - started
        self._stream_log(
            f"done in {total_elapsed:.2f}s; chunks={chunk_count}; tool_calls={len(tool_calls)}"
        )
        return data

    def next_tool_calls(
        self,
        *,
        state_text: str,
        tool_schemas: list[dict[str, Any]],
        instructions: str,
        state_image: dict[str, Any] | None = None,
    ) -> ProviderResult:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            return ProviderResult(
                [],
                raw=None,
                error=_missing_openai_dependency_message(exc),
            )

        headers = {}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers or None,
            timeout=self.timeout_s,
        )

        tools = _tool_schemas_to_openai_chat(tool_schemas)
        if state_image:
            content: list[dict[str, Any]] = []
            if state_text:
                content.append({"type": "text", "text": state_text})
            data_url = state_image.get("data_url")
            if not data_url:
                mime = state_image.get("mime_type", "image/png")
                data = state_image.get("data_base64")
                if data:
                    data_url = f"data:{mime};base64,{data}"
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": content or state_text},
            ]
        else:
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": state_text},
            ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        attempt = 0
        data: dict[str, Any]
        while True:
            try:
                data = self._completion_payload(client, kwargs)
                break
            except Exception as exc:  # pragma: no cover
                attempt += 1
                if attempt > self.max_retries:
                    return ProviderResult([], raw=None, error=f"Provider error: {exc}")
                time.sleep(self.retry_backoff_s * (2 ** (attempt - 1)))
        usage = _normalize_usage(data.get("usage"))
        cost = _extract_cost(usage, data)
        choices = data.get("choices", [])
        if not choices:
            return ProviderResult(
                [], raw=data, error="No choices returned.", usage=usage, cost=cost
            )
        message = choices[0].get("message", {})
        err = None
        if isinstance(choices[0], dict) and "error" in choices[0]:
            err = choices[0].get("error")
        if err and isinstance(err, dict):
            code = err.get("code")
            if isinstance(code, int) and 500 <= code < 600:
                # Retry on upstream 5xx embedded in payload.
                if self.max_retries < 1:
                    return ProviderResult(
                        [],
                        raw=data,
                        error=f"Provider error: {err}",
                        usage=usage,
                        cost=cost,
                    )
                retry_exception: Exception | None = None
                retry_error_payload: dict[str, Any] | None = err
                recovered = False
                for attempt in range(1, self.max_retries + 1):
                    time.sleep(self.retry_backoff_s * (2 ** (attempt - 1)))
                    try:
                        data = self._completion_payload(client, kwargs)
                        retry_exception = None
                        usage = _normalize_usage(data.get("usage"))
                        cost = _extract_cost(usage, data)
                        choices = data.get("choices", [])
                        if not choices:
                            return ProviderResult(
                                [],
                                raw=data,
                                error="No choices returned.",
                                usage=usage,
                                cost=cost,
                            )
                        message = choices[0].get("message", {})
                        retry_error_payload = (
                            choices[0].get("error")
                            if isinstance(choices[0], dict)
                            else None
                        )
                        retry_code = (
                            retry_error_payload.get("code")
                            if isinstance(retry_error_payload, dict)
                            else None
                        )
                        if isinstance(retry_code, int) and 500 <= retry_code < 600:
                            continue
                        recovered = True
                        break
                    except Exception as exc:
                        retry_exception = exc
                        continue
                if not recovered:
                    if retry_exception is not None:
                        return ProviderResult(
                            [],
                            raw=data,
                            error=f"Provider error after retries: {retry_exception}",
                            usage=usage,
                            cost=cost,
                        )
                    return ProviderResult(
                        [],
                        raw=data,
                        error=f"Provider error after retries: {retry_error_payload}",
                        usage=usage,
                        cost=cost,
                    )
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return ProviderResult(
                _normalize_tool_calls(tool_calls), raw=data, usage=usage, cost=cost
            )
        if "function_call" in message:
            return ProviderResult(
                _normalize_tool_calls(message["function_call"]),
                raw=data,
                usage=usage,
                cost=cost,
            )
        return ProviderResult(
            [], raw=data, error="No tool calls returned.", usage=usage, cost=cost
        )


class OpenAIResponsesProvider:
    """OpenAI Responses API provider."""

    supports_images = False

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def next_tool_calls(
        self,
        *,
        state_text: str,
        tool_schemas: list[dict[str, Any]],
        instructions: str,
        state_image: dict[str, Any] | None = None,
    ) -> ProviderResult:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            return ProviderResult(
                [],
                raw=None,
                error=_missing_openai_dependency_message(exc),
            )

        client = OpenAI(api_key=self.api_key)
        tools = _tool_schemas_to_openai_responses(tool_schemas)
        if state_image:
            return ProviderResult(
                [],
                raw=None,
                error="Image inputs are not supported by this provider in this harness.",
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "tools": tools,
            "tool_choice": "required",
            "input": [{"role": "user", "content": state_text}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens

        response = client.responses.create(**kwargs)
        data = response.model_dump() if hasattr(response, "model_dump") else response
        usage = None
        if isinstance(data, dict):
            usage = _normalize_usage(data.get("usage"))
        if usage is None and hasattr(response, "usage"):
            usage = _normalize_usage(getattr(response, "usage"))
        cost = _extract_cost(usage, data if isinstance(data, dict) else None)

        output = getattr(response, "output", None)
        output_items = output if output is not None else []
        tool_calls = _normalize_tool_calls(output_items)
        if not tool_calls and isinstance(data, dict):
            tool_calls = _normalize_tool_calls(data.get("output"))
        if tool_calls:
            return ProviderResult(
                tool_calls,
                raw=response,
                usage=usage,
                cost=cost,
            )
        return ProviderResult(
            [], raw=response, error="No tool calls returned.", usage=usage, cost=cost
        )


class CLIProvider:
    """Run a local CLI and parse a JSON tool call from stdout."""

    supports_images = False

    def __init__(
        self,
        *,
        command: str | Iterable[str],
        use_stdin: bool = True,
        timeout_s: int = 120,
    ) -> None:
        self.command = command
        self.use_stdin = use_stdin
        self.timeout_s = timeout_s

    def _build_command(self, prompt: str) -> list[str]:
        if isinstance(self.command, str):
            if "{prompt}" in self.command:
                rendered = self.command.format(prompt=prompt)
                return shlex.split(rendered)
            base = shlex.split(self.command)
        else:
            base = [str(x) for x in self.command]
            if any("{prompt}" in item for item in base):
                return [item.format(prompt=prompt) for item in base]
        if self.use_stdin:
            return base
        return base + [prompt]

    def _uses_prompt_placeholder(self) -> bool:
        if isinstance(self.command, str):
            return "{prompt}" in self.command
        return any("{prompt}" in str(item) for item in self.command)

    def next_tool_calls(
        self,
        *,
        state_text: str,
        tool_schemas: list[dict[str, Any]],
        instructions: str,
        state_image: dict[str, Any] | None = None,
    ) -> ProviderResult:
        if state_image:
            return ProviderResult(
                [],
                raw=None,
                error="Image inputs are not supported by CLIProvider.",
            )
        prompt = (
            f"{instructions}\n\nSTATE:\n{state_text}\n\nTOOLS:\n"
            f"{json.dumps(tool_schemas, indent=2)}\n\n"
            "Return a single tool call as JSON: "
            '{"name": "...", "arguments": {"from_peg": 0, "to_peg": 2}}'
        )
        cmd = self._build_command(prompt)
        use_stdin = self.use_stdin and not self._uses_prompt_placeholder()
        try:
            completed = subprocess.run(
                cmd,
                input=prompt if use_stdin else None,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ProviderResult([], raw=None, error="CLI timed out.")

        output = completed.stdout.strip() or completed.stderr.strip()
        parsed = _parse_json_loose(output)
        tool_calls = _normalize_tool_calls(parsed if parsed is not None else output)
        if tool_calls:
            return ProviderResult(tool_calls, raw=output)
        return ProviderResult(
            [], raw=output, error="No tool calls parsed from CLI output."
        )


class CodexCLIProvider:
    """Codex CLI provider using --output-last-message and --output-schema."""

    supports_images = False

    def __init__(
        self,
        *,
        codex_path: str = "codex",
        extra_args: Iterable[str] | None = None,
        timeout_s: int = 300,
    ) -> None:
        self.codex_path = codex_path
        self.extra_args = list(extra_args or [])
        self.timeout_s = timeout_s
        self._tmpdir = tempfile.TemporaryDirectory()
        self._schema_path = Path(self._tmpdir.name) / "tool_call.schema.json"
        self._schema_path.write_text(
            json.dumps(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
                indent=2,
            )
        )

    def next_tool_calls(
        self,
        *,
        state_text: str,
        tool_schemas: list[dict[str, Any]],
        instructions: str,
        state_image: dict[str, Any] | None = None,
    ) -> ProviderResult:
        if state_image:
            return ProviderResult(
                [],
                raw=None,
                error="Image inputs are not supported by CodexCLIProvider.",
            )
        prompt = (
            f"{instructions}\n\nSTATE:\n{state_text}\n\nTOOLS:\n"
            f"{json.dumps(tool_schemas, indent=2)}\n\n"
            "Return a single tool call as JSON: "
            '{"name": "...", "arguments": {"from_peg": 0, "to_peg": 2}}'
        )
        out_path = Path(self._tmpdir.name) / "last_message.json"
        if out_path.exists():
            out_path.unlink()

        cmd = [
            self.codex_path,
            "exec",
            "--output-last-message",
            str(out_path),
            "--output-schema",
            str(self._schema_path),
            *self.extra_args,
            prompt,
        ]
        try:
            completed = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ProviderResult([], raw=None, error="Codex CLI timed out.")

        if completed.returncode != 0:
            return ProviderResult(
                [],
                raw=completed.stderr.strip() or completed.stdout.strip(),
                error="Codex CLI returned a non-zero exit code.",
            )
        if not out_path.exists():
            return ProviderResult(
                [], raw=None, error="Codex CLI did not write output-last-message."
            )
        output = out_path.read_text().strip()
        parsed = _parse_json_loose(output)
        tool_calls = _normalize_tool_calls(parsed if parsed is not None else output)
        if tool_calls:
            return ProviderResult(tool_calls, raw=output)
        return ProviderResult(
            [], raw=output, error="No tool calls parsed from Codex output."
        )
