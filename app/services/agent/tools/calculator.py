import ast
import operator
from collections.abc import Callable
from typing import Any

from app.services.agent.tools.base import BaseTool, ToolResult

_SAFE_OPERATORS: dict[type[ast.AST], Callable[..., float | int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node containing only safe arithmetic operations."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        bin_op_type = type(node.op)
        if bin_op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported mathematical operator: {bin_op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if bin_op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero is undefined")
        result = _SAFE_OPERATORS[bin_op_type](left, right)
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result
    elif isinstance(node, ast.UnaryOp):
        unary_op_type = type(node.op)
        if unary_op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {unary_op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        unary_result = _SAFE_OPERATORS[unary_op_type](operand)
        if isinstance(unary_result, float) and unary_result.is_integer():
            return int(unary_result)
        return unary_result
    else:
        raise ValueError(f"Unsupported mathematical expression construct: {type(node).__name__}")


def safe_calculate(expression: str) -> float | int:
    """Safely parse and evaluate a mathematical expression string using AST."""
    cleaned = expression.strip()
    if not cleaned:
        raise ValueError("Expression cannot be empty")
    parsed = ast.parse(cleaned, mode="eval")
    return _eval_ast_node(parsed)


class CalculatorTool(BaseTool):
    """Tool for evaluating safe mathematical expressions and arithmetic calculations."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Perform mathematical calculations and evaluate arithmetic expressions "
            "(e.g. '25 * 4 + 10', '1024 / 8', '2 ** 10')."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression string to evaluate",
                }
            },
            "required": ["expression"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        expression = kwargs.get("expression")
        if not expression or not isinstance(expression, str):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "Missing or invalid required argument: 'expression' "
                    "(must be a non-empty string)"
                ),
            )

        try:
            result = safe_calculate(expression)
            # Format integer outputs without floating point suffix
            formatted = int(result) if isinstance(result, float) and result.is_integer() else result
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=formatted,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Calculation error: {exc}",
            )
