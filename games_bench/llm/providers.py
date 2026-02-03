from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
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
        if "tool_call" in obj:
            return _normalize_tool_calls(obj["tool_call"])
        if "tool_calls" in obj:
            return _normalize_tool_calls(obj["tool_calls"])
        if "name" in obj and "arguments" in obj:
            args = obj["arguments"]
            if isinstance(args, str):
                parsed = _parse_json_loose(args)
                if isinstance(parsed, dict):
                    args = parsed
            if not isinstance(args, dict):
                args = {"value": args}
            return [ToolCall(name=str(obj["name"]), arguments=args)]
        if "function" in obj and isinstance(obj["function"], dict):
            func = obj["function"]
            if "name" in func and "arguments" in func:
                args = func["arguments"]
                if isinstance(args, str):
                    parsed = _parse_json_loose(args)
                    if isinstance(parsed, dict):
                        args = parsed
                if not isinstance(args, dict):
                    args = {"value": args}
                return [ToolCall(name=str(func["name"]), arguments=args)]
        for key in ("content", "message", "output", "text"):
            if key in obj:
                return _normalize_tool_calls(obj[key])
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


class OpenRouterProvider:
    """OpenRouter provider using OpenAI-compatible Chat Completions."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY")
        self.http_referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        self.x_title = x_title or os.environ.get("OPENROUTER_X_TITLE")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def next_tool_calls(
        self,
        *,
        state_text: str,
        tool_schemas: list[dict[str, Any]],
        instructions: str,
    ) -> ProviderResult:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            return ProviderResult([], raw=None, error=f"Missing dependency: {exc}")

        headers = {}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers or None,
        )

        tools = _tool_schemas_to_openai_chat(tool_schemas)
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

        response = client.chat.completions.create(**kwargs)
        data = response.model_dump() if hasattr(response, "model_dump") else response
        usage = _normalize_usage(data.get("usage")) if isinstance(data, dict) else None
        cost = _extract_cost(usage, data if isinstance(data, dict) else None)
        choices = data.get("choices", [])
        if not choices:
            return ProviderResult(
                [], raw=data, error="No choices returned.", usage=usage, cost=cost
            )
        message = choices[0].get("message", {})
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
    ) -> ProviderResult:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            return ProviderResult([], raw=None, error=f"Missing dependency: {exc}")

        client = OpenAI(api_key=self.api_key)
        tools = _tool_schemas_to_openai_responses(tool_schemas)
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
        output_items = output or []
        tool_calls = [
            x for x in output_items if getattr(x, "type", None) == "function_call"
        ]
        if tool_calls:
            return ProviderResult(
                _normalize_tool_calls(tool_calls),
                raw=response,
                usage=usage,
                cost=cost,
            )
        return ProviderResult(
            [], raw=response, error="No tool calls returned.", usage=usage, cost=cost
        )


class CLIProvider:
    """Run a local CLI and parse a JSON tool call from stdout."""

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
    ) -> ProviderResult:
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
    ) -> ProviderResult:
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
