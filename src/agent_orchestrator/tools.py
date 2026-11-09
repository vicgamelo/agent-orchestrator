"""Tools: anything an agent can call. Python functions, MCP tools and other agents
(delegation) all end up as the same :class:`Tool`."""

from __future__ import annotations

import ast
import inspect
import json
import math
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter, create_model

from .llm.base import ToolSpec

ToolFn = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    source: str = "python"  # python | mcp:<server> | agent

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    async def __call__(self, arguments: dict[str, Any]) -> str:
        return await self.fn(arguments)


def function_tool(fn: Callable[..., Any], *, name: str | None = None, description: str | None = None) -> Tool:
    """Build a tool from a typed Python function. The JSON schema comes from the
    signature via pydantic, and arguments are validated before the call."""
    sig = inspect.signature(fn)
    fields: dict[str, Any] = {
        p.name: (p.annotation, ... if p.default is inspect.Parameter.empty else p.default)
        for p in sig.parameters.values()
    }
    model = create_model(f"{fn.__name__}_args", **fields)
    schema = model.model_json_schema()
    schema.pop("title", None)

    async def call(arguments: dict[str, Any]) -> str:
        args = model.model_validate(arguments)
        result = fn(**dict(args))
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return result
        return json.dumps(TypeAdapter(Any).dump_python(result, mode="json"), ensure_ascii=False)

    return Tool(
        name=name or fn.__name__,
        description=description or inspect.getdoc(fn) or fn.__name__,
        parameters=schema,
        fn=call,
    )


# ----------------------------------------------------------------------------
# Built-in tools

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without ``eval``: only numbers, + - * / // % **, and a
    whitelist of math functions. Everything is a float: Python ints have no size limit,
    so nested powers like ``((9**99)**99)**99`` would pin the CPU; floats overflow at once."""

    def ev(node: ast.AST) -> Any:
        match node:
            case ast.Expression(body=body):
                return ev(body)
            case ast.Constant(value=v) if isinstance(v, int | float) and not isinstance(v, bool):
                return float(v)
            case ast.BinOp(left=l, op=op, right=r) if type(op) in _BIN_OPS:
                left, right = float(ev(l)), float(ev(r))
                if isinstance(op, ast.Pow) and abs(right) > 100:
                    raise ValueError("exponent too large")
                return _BIN_OPS[type(op)](left, right)
            case ast.UnaryOp(op=op, operand=o) if type(op) in _UNARY_OPS:
                return _UNARY_OPS[type(op)](float(ev(o)))
            case ast.Call(func=ast.Name(id=fname), args=args, keywords=[]) if fname in _FUNCS:
                return float(_FUNCS[fname](*[float(ev(a)) for a in args]))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")

    if len(expression) > 300:
        raise ValueError("expression too long")
    return float(ev(ast.parse(expression, mode="eval")))


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(1.0125 ** 12 - 1) * 100'. Supports + - * / // % **
    and sqrt, log, exp, abs, round, min, max."""
    try:
        value = safe_eval(expression)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return f"error: {exc}"
    return f"{value:.10g}"


def current_datetime(timezone: str = "America/Sao_Paulo") -> str:
    """Current date and time in ISO 8601 for an IANA timezone."""
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")


BUILTIN_TOOLS: dict[str, Callable[[], Tool]] = {
    "calculator": lambda: function_tool(calculator),
    "current_datetime": lambda: function_tool(current_datetime),
}
