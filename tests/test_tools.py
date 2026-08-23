from app.config import Settings
from app.services.agent.tools.app_info import AppInfoTool
from app.services.agent.tools.calculator import CalculatorTool, safe_calculate
from app.services.agent.tools.registry import ToolRegistry


def test_safe_calculate_evaluations() -> None:
    """Verify safe calculator handles basic arithmetic expressions."""
    assert safe_calculate("25 * 4") == 100
    assert safe_calculate("10 + 20 * 3") == 70
    assert safe_calculate("(10 + 20) * 3") == 90
    assert safe_calculate("1024 / 8") == 128
    assert safe_calculate("2 ** 8") == 256
    assert safe_calculate("17 % 5") == 2
    assert safe_calculate("-5 + 10") == 5


def test_safe_calculate_division_by_zero() -> None:
    """Verify safe calculator captures division by zero."""
    tool = CalculatorTool()
    result = tool.execute(expression="100 / 0")
    assert not result.success
    assert "Division by zero" in (result.error or "")


def test_safe_calculate_invalid_expression() -> None:
    """Verify safe calculator rejects invalid syntax or unsafe operations."""
    tool = CalculatorTool()
    result = tool.execute(expression="import os")
    assert not result.success
    assert result.error is not None


def test_calculator_tool_missing_arg() -> None:
    """Verify calculator handles missing or empty arguments."""
    tool = CalculatorTool()
    res1 = tool.execute()
    assert not res1.success
    assert "Missing or invalid required argument" in (res1.error or "")

    res2 = tool.execute(expression="")
    assert not res2.success


def test_app_info_tool() -> None:
    """Verify AppInfoTool returns application metadata from settings."""
    settings = Settings(
        app_name="Custom Test App",
        app_version="1.2.3",
        app_env="staging",
    )
    tool = AppInfoTool(settings=settings)
    assert tool.name == "app_info"
    result = tool.execute()

    assert result.success
    assert result.output["app_name"] == "Custom Test App"
    assert result.output["version"] == "1.2.3"
    assert result.output["environment"] == "staging"


def test_tool_registry_registration_and_lookup() -> None:
    """Verify ToolRegistry registers, retrieves, and lists tools."""
    registry = ToolRegistry()
    assert "calculator" in registry.available_tool_names
    assert "app_info" in registry.available_tool_names

    calc = registry.get("calculator")
    assert calc is not None
    assert calc.name == "calculator"

    # Case-insensitive lookup
    assert registry.get("CALCULATOR") is not None


def test_tool_registry_unknown_tool_execution() -> None:
    """Verify ToolRegistry handles unknown tool names gracefully."""
    registry = ToolRegistry()
    result = registry.execute(tool_name="non_existent_tool")
    assert not result.success
    assert "not found" in (result.error or "")
