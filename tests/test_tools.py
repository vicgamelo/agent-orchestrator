import pytest

from agent_orchestrator.tools import calculator, function_tool, safe_eval

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("expr", "value"),
    [("1 + 2 * 3", 7), ("(1.0125 ** 12 - 1) * 100", 16.075451772299854), ("sqrt(16) + max(1, 5)", 9), ("-3 // 2", -2)],
)
def test_safe_eval(expr: str, value: float) -> None:
    assert safe_eval(expr) == pytest.approx(value)


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('ls')", "open('/etc/passwd')", "2 ** 100000", "x + 1", "(lambda: 1)()", "True + 1"],
)
def test_safe_eval_rejects_everything_else(expr: str) -> None:
    assert calculator(expr).startswith("error:")


def test_division_by_zero_is_reported() -> None:
    assert calculator("1/0").startswith("error:")


async def test_function_tool_schema_and_validation() -> None:
    def add(a: int, b: int = 2) -> dict[str, int]:
        """Add two numbers."""
        return {"sum": a + b}

    tool = function_tool(add)
    assert tool.description == "Add two numbers."
    assert tool.parameters["required"] == ["a"]
    assert tool.parameters["properties"]["b"]["default"] == 2
    assert await tool({"a": "3"}) == '{"sum": 5}'
    with pytest.raises(ValueError):
        await tool({"a": "not a number"})


@pytest.mark.parametrize(
    "expr",
    [
        "(((9**99)**99)**99)**99",  # cada nível multiplica o expoente; com int do Python isso trava a CPU
        "((((2**100)**100)**100)**100)**100",
        "(10**100)**100 * (10**100)**100",
    ],
)
def test_nested_powers_fail_fast_instead_of_hanging(expr: str) -> None:
    # Roda num processo separado: sem a correção ele fica preso por minutos e o timeout derruba o teste.
    import subprocess
    import sys

    code = f"from agent_orchestrator.tools import calculator; print(calculator({expr!r}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert out.stdout.startswith("error:"), out.stdout + out.stderr


def test_float_math_still_correct() -> None:
    assert safe_eval("(1.0125 ** 12 - 1) * 100") == pytest.approx(16.0755, abs=1e-4)
    assert safe_eval("7 // 2") == 3
    assert safe_eval("2 ** 10") == 1024
