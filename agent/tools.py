"""
Tool definitions for the agent.

Each tool has two parts:
1. A JSON schema (so the LLM knows the tool exists and how to call it)
2. A Python function that actually executes it

This is the pattern every agent framework (LangGraph, OpenAI Assistants,
Anthropic's tool use) is built on top of — they just add orchestration
and syntactic sugar around this exact idea.
"""

import ast
import operator
import os

# Tools are only allowed to read/write inside this sandboxed folder.
# Never let an LLM-controlled agent write anywhere on the real filesystem.
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Tool 1: Calculator
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    """Evaluate a parsed arithmetic expression without using eval()."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# Tool 2: File write (sandboxed)
# ---------------------------------------------------------------------------

def write_file(filename: str, content: str) -> str:
    # Prevent path traversal outside the workspace
    safe_name = os.path.basename(filename)
    path = os.path.join(WORKSPACE_DIR, safe_name)
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {safe_name}"


# ---------------------------------------------------------------------------
# Tool 3: File read (sandboxed)
# ---------------------------------------------------------------------------

def read_file(filename: str) -> str:
    safe_name = os.path.basename(filename)
    path = os.path.join(WORKSPACE_DIR, safe_name)
    if not os.path.exists(path):
        return f"Error: {safe_name} does not exist in workspace"
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Tool registry: schema (sent to the model) + function (executed locally)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "schema": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression (+, -, *, /, **, %). "
                            "Use this for any math instead of doing it yourself.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '23 * 4 + 1'",
                    }
                },
                "required": ["expression"],
            },
        },
        "function": calculator,
    },
    {
        "schema": {
            "name": "write_file",
            "description": "Write text content to a file in the agent's workspace directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file, e.g. 'notes.txt'"},
                    "content": {"type": "string", "description": "The text content to write"},
                },
                "required": ["filename", "content"],
            },
        },
        "function": write_file,
    },
    {
        "schema": {
            "name": "read_file",
            "description": "Read text content from a file in the agent's workspace directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file to read"},
                },
                "required": ["filename"],
            },
        },
        "function": read_file,
    },
]

TOOL_SCHEMAS = [t["schema"] for t in TOOLS]
TOOL_FUNCTIONS = {t["schema"]["name"]: t["function"] for t in TOOLS}
