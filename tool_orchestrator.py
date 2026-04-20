"""
tool_orchestrator.py
FreshMart Tool Orchestrator

Registers all 4 tools, detects tool-call JSON in LLM output,
executes the right tool asynchronously, and returns the result.

LLM tool-call format (embedded anywhere in the response):
    <tool_call>{"tool": "get_weather", "args": {"city": "Rawalpindi"}}</tool_call>

The orchestrator strips this from the response, runs the tool,
then re-prompts the LLM with the result to produce a natural reply.
"""

import re
import json
import asyncio
import logging
from typing import Optional

from crm_tool import (
    get_user_info,
    update_user_info,
    store_interaction,
    get_interaction_history,
    CRM_TOOL_SCHEMAS,
)
from tools.weather_tool   import get_weather,        WEATHER_TOOL_SCHEMA
from tools.currency_tool  import convert_currency,   CURRENCY_TOOL_SCHEMA
from tools.calculator_tool import calculate,          CALCULATOR_TOOL_SCHEMA

logger = logging.getLogger(__name__)

# ── Tool Registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    # CRM tools (sync)
    "get_user_info":           ("sync",  get_user_info),
    "update_user_info":        ("sync",  update_user_info),
    "store_interaction":       ("sync",  store_interaction),
    "get_interaction_history": ("sync",  get_interaction_history),
    # Additional tools
    "get_weather":             ("async", get_weather),
    "convert_currency":        ("async", convert_currency),
    "calculate":               ("sync",  calculate),
}

ALL_TOOL_SCHEMAS = CRM_TOOL_SCHEMAS + [
    WEATHER_TOOL_SCHEMA,
    CURRENCY_TOOL_SCHEMA,
    CALCULATOR_TOOL_SCHEMA,
]

# Regex to find tool calls in LLM output
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL | re.IGNORECASE
)

TOOL_TIMEOUT = 8.0   # seconds before a tool call is considered failed


# ── Core Functions ────────────────────────────────────────────────────────────

def extract_tool_call(text: str) -> Optional[dict]:
    """
    Find and parse the first <tool_call>...</tool_call> block in text.
    Returns the parsed dict or None if not found.
    """
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        logger.warning(f"[Orchestrator] Failed to parse tool call JSON: {e}")
        return None


def strip_tool_call(text: str) -> str:
    """Remove all <tool_call>...</tool_call> blocks from text."""
    return TOOL_CALL_PATTERN.sub("", text).strip()


async def execute_tool(tool_name: str, args: dict) -> dict:
    """
    Execute a registered tool by name with the given arguments.
    Handles both sync and async tools with timeout protection.

    Returns:
        dict result from the tool, or an error dict on failure.
    """
    if tool_name not in TOOL_REGISTRY:
        available = ", ".join(TOOL_REGISTRY.keys())
        return {"error": f"Unknown tool '{tool_name}'. Available: {available}"}

    kind, func = TOOL_REGISTRY[tool_name]

    try:
        if kind == "async":
            result = await asyncio.wait_for(func(**args), timeout=TOOL_TIMEOUT)
        else:
            # Run sync tools in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(**args)),
                timeout=TOOL_TIMEOUT
            )
        logger.info(f"[Orchestrator] Tool '{tool_name}' executed successfully.")
        return result

    except asyncio.TimeoutError:
        logger.warning(f"[Orchestrator] Tool '{tool_name}' timed out after {TOOL_TIMEOUT}s")
        return {"error": f"Tool '{tool_name}' timed out. Please try again."}
    except TypeError as e:
        logger.error(f"[Orchestrator] Tool '{tool_name}' bad arguments: {e}")
        return {"error": f"Invalid arguments for tool '{tool_name}': {str(e)}"}
    except Exception as e:
        logger.error(f"[Orchestrator] Tool '{tool_name}' error: {e}")
        return {"error": f"Tool '{tool_name}' failed: {str(e)}"}


async def process_tool_call(llm_response: str) -> tuple[str, Optional[dict]]:
    """
    Check an LLM response for a tool call, execute it if found.

    Returns:
        (clean_response, tool_result)
        - clean_response: LLM text with tool_call tags stripped
        - tool_result: dict result from the tool, or None if no tool call found
    """
    tool_call = extract_tool_call(llm_response)
    clean_text = strip_tool_call(llm_response)

    if tool_call is None:
        return clean_text, None

    tool_name = tool_call.get("tool", "")
    args      = tool_call.get("args", {})

    if not tool_name:
        return clean_text, {"error": "Tool call missing 'tool' field"}

    logger.info(f"[Orchestrator] Calling tool: {tool_name} with args: {args}")
    result = await execute_tool(tool_name, args)
    return clean_text, result


def format_tool_schemas_for_prompt() -> str:
    """
    Format all tool schemas as a string to inject into the system prompt.
    Teaches the LLM how to call tools.
    """
    lines = ["[AVAILABLE TOOLS]",
             "You can call the following tools by outputting a <tool_call> block.",
             "Format: <tool_call>{\"tool\": \"tool_name\", \"args\": {\"param\": \"value\"}}</tool_call>",
             "Only call ONE tool at a time. Only call a tool when it is genuinely needed.",
             ""]

    for schema in ALL_TOOL_SCHEMAS:
        params = ", ".join(
            f"{k} ({v.get('type','str')}): {v.get('description','')}"
            for k, v in schema.get("parameters", {}).items()
        )
        lines.append(f"Tool: {schema['name']}")
        lines.append(f"  Description: {schema['description']}")
        lines.append(f"  Parameters:  {params}")
        lines.append("")

    lines.append("[END OF TOOLS]")
    return "\n".join(lines)


def format_tool_result_for_prompt(tool_name: str, result: dict) -> str:
    """Format a tool result to inject back into the LLM conversation."""
    result_str = json.dumps(result, indent=2)
    return (
        f"[TOOL RESULT: {tool_name}]\n"
        f"{result_str}\n"
        f"[END TOOL RESULT]\n"
        f"Based on the above tool result, provide a helpful and natural response to the user."
    )


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def test():
        print("Testing Tool Orchestrator...\n")

        # Test 1: calculator
        print("Test 1 - Calculator via tool call:")
        fake_llm = 'Let me calculate that. <tool_call>{"tool": "calculate", "args": {"expression": "3 * 2.50 + 1.80"}}</tool_call>'
        clean, result = await process_tool_call(fake_llm)
        print(f"  Clean text: '{clean}'")
        print(f"  Result: {result}")

        # Test 2: CRM
        print("\nTest 2 - CRM get_user_info:")
        fake_llm2 = '<tool_call>{"tool": "get_user_info", "args": {"user_id": "demo_session"}}</tool_call>'
        clean2, result2 = await process_tool_call(fake_llm2)
        print(f"  Result: {result2}")

        # Test 3: unknown tool
        print("\nTest 3 - Unknown tool (error handling):")
        result3 = await execute_tool("fly_to_moon", {})
        print(f"  Result: {result3}")

        # Test 4: tool schemas
        print("\nTest 4 - Tool schemas for prompt:")
        schemas = format_tool_schemas_for_prompt()
        print(schemas[:300] + "...")

    asyncio.run(test())
