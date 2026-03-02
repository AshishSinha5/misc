"""
Math Agent — breaks complex math problems into atomic operations using an LLM brain.
LLM is accessed via OpenRouter (OpenAI-compatible API).
"""

import os
import json
import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
MODEL = os.environ["OPENROUTER_MODEL"]

# ---------------------------------------------------------------------------
# Math tools (pure functions the LLM can call)
# ---------------------------------------------------------------------------

def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

def power(base: float, exp: float) -> float:
    return base ** exp

def sqrt(x: float) -> float:
    if x < 0:
        raise ValueError("sqrt of negative number")
    return math.sqrt(x)

def log(x: float, base: float = math.e) -> float:
    if x <= 0:
        raise ValueError("log of non-positive number")
    return math.log(x, base)

def factorial(n: int) -> int:
    if n < 0 or not float(n).is_integer():
        raise ValueError("factorial requires a non-negative integer")
    return math.factorial(int(n))

def modulo(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Modulo by zero")
    return a % b

def sin_deg(degrees: float) -> float:
    return math.sin(math.radians(degrees))

def cos_deg(degrees: float) -> float:
    return math.cos(math.radians(degrees))

def tan_deg(degrees: float) -> float:
    return math.tan(math.radians(degrees))

def absolute(x: float) -> float:
    return abs(x)

def floor(x: float) -> int:
    return math.floor(x)

def ceil(x: float) -> int:
    return math.ceil(x)

# Map name → callable
TOOLS: dict = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
    "power": power,
    "sqrt": sqrt,
    "log": log,
    "factorial": factorial,
    "modulo": modulo,
    "sin_deg": sin_deg,
    "cos_deg": cos_deg,
    "tan_deg": tan_deg,
    "absolute": absolute,
    "floor": floor,
    "ceil": ceil,
}

# ---------------------------------------------------------------------------
# OpenAI-format tool schemas
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers: a + b",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Subtract two numbers: a - b",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers: a * b",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Divide a by b: a / b",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number", "description": "Divisor (must not be 0)"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "power",
            "description": "Raise base to exponent: base ** exp",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "number"},
                    "exp": {"type": "number"},
                },
                "required": ["base", "exp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sqrt",
            "description": "Square root of x",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Non-negative number"},
                },
                "required": ["x"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log",
            "description": "Logarithm of x. Defaults to natural log (base e). Pass base=10 for log10.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "base": {"type": "number", "description": "Log base (default: e)"},
                },
                "required": ["x"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "factorial",
            "description": "Factorial of a non-negative integer n",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "number"},
                },
                "required": ["n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modulo",
            "description": "Remainder of a divided by b: a % b",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sin_deg",
            "description": "Sine of an angle in degrees",
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {"type": "number"},
                },
                "required": ["degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cos_deg",
            "description": "Cosine of an angle in degrees",
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {"type": "number"},
                },
                "required": ["degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tan_deg",
            "description": "Tangent of an angle in degrees",
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {"type": "number"},
                },
                "required": ["degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "absolute",
            "description": "Absolute value of x",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                },
                "required": ["x"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "floor",
            "description": "Floor of x (round down to nearest integer)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                },
                "required": ["x"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ceil",
            "description": "Ceiling of x (round up to nearest integer)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                },
                "required": ["x"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise math agent. When given a math problem:
1. Break it into the smallest possible atomic operations.
2. Call the appropriate tool for each operation.
3. Use the result of each tool call as input to the next step.
4. After all calculations are done, respond with a clear final answer.

Never guess or approximate — always use tools for every numeric computation.
"""

def run_agent(problem: str, max_iterations: int = 20) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    print(f"\nProblem: {problem}\n{'─' * 50}")

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Append the assistant message (with any tool calls) to history
        messages.append(msg)

        # No more tool calls — the agent is done
        if finish_reason == "stop" or not msg.tool_calls:
            print(f"\nFinal Answer: {msg.content}")
            return msg.content

        # Execute each requested tool call
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if fn_name not in TOOLS:
                result = f"Error: unknown tool '{fn_name}'"
            else:
                try:
                    result = TOOLS[fn_name](**fn_args)
                    print(f"  [{iteration + 1}] {fn_name}({fn_args}) = {result}")
                except Exception as exc:
                    result = f"Error: {exc}"
                    print(f"  [{iteration + 1}] {fn_name}({fn_args}) => {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    return "Agent hit max iterations without a final answer."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    problems = [
        "What is (3! + sqrt(144)) * 2 - log(1000, 10)?",
        "A triangle has legs of length 5 and 12. What is the hypotenuse? Then find sin of the angle opposite the leg of length 5 (in degrees).",
        "Compute: (2^10 - 1) / 3, then take the floor, then find its factorial mod 7.",
    ]

    for p in problems:
        run_agent(p)
        print()
