import json
import re
import uuid
import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletion
from mcp import ClientSession
from mcp.types import CallToolResult
from .data_functions import read_student_csv, read_student_csv_tool

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
MAX_TOOL_ROUNDS = 6  # safety limit so the model can't loop forever

# 1. Keep your local function mapping intact
LOCAL_FUNCTION_REGISTRY = {
    "read_student_csv": read_student_csv
}

LOCAL_TOOLS = [
    read_student_csv_tool,
]


# 2. Hybrid Function Executor (async to handle MCP calls)
async def execute_tool(func_name, func_args, mcp_session: ClientSession = None):
    # ROUTE A: Execute locally if it exists in your registry
    if func_name in LOCAL_FUNCTION_REGISTRY:
        return LOCAL_FUNCTION_REGISTRY[func_name](**func_args)

    # ROUTE B: Route over the network to Supabase if an active MCP session exists
    elif mcp_session is not None:
        try:
            mcp_result = await mcp_session.call_tool(name=func_name, arguments=func_args)
            return mcp_result.content
        except Exception as e:
            return {"status": "error", "content": f"Supabase MCP execution failed: {str(e)}"}
            
    else:
        return {"status": "error", "content": f"No function or MCP server found for {func_name}!"}

#* tool results to text.
def _block_to_text(block) -> str:
    text = getattr(block, "text", None)
    if text is not None:
        return text
    if hasattr(block, "model_dump"):
        return json.dumps(block.model_dump(mode="json"))
    return str(block)

def tool_result_to_text(result) -> str:
    # Full MCP result object
    if isinstance(result, CallToolResult):
        text = "\n".join(_block_to_text(b) for b in (result.content or []))
        return f"Tool error: {text}" if result.isError else text

    # MCP content list
    if isinstance(result, list) and any(hasattr(b, "type") and hasattr(b, "model_dump") for b in result):
        return "\n".join(_block_to_text(b) for b in result)

    # Local tool results
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except TypeError:
        return str(result)


# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------
def _assistant_tool_message(tools_called, content=None):
    """Build the assistant message that records the tool calls in the history."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": t["id"],
            "type": "function",
            "function": {
                "name": t["func_name"],
                "arguments": json.dumps(t["func_args"]),
            },
        } for t in tools_called],
    }


def process_tool_call(response: ChatCompletion):
    """Extract proper (structured) tool calls from the model response."""
    message = response.choices[0].message
    if not getattr(message, "tool_calls", None):
        return None, None

    tools_called = []
    for tool in message.tool_calls:
        try:
            args = json.loads(tool.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        tools_called.append({
            "id": tool.id,
            "func_name": tool.function.name,
            "func_args": args,
        })
    return tools_called, _assistant_tool_message(tools_called, message.content)


# Some models/servers return their native tool call markup as plain text, like:
#   <tool_call><function=execute_sql><parameter=query>SELECT ...</parameter></function></tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def parse_text_tool_calls(content):
    """Fallback: turn tool call markup found in message text into real tool calls."""
    calls = []
    if not content or "<tool_call>" not in content:
        return calls
    for block in _TOOL_CALL_RE.findall(content):
        fn = _FUNCTION_RE.search(block)
        if not fn:
            continue
        name, body = fn.group(1), fn.group(2)
        args = {key: value.strip() for key, value in _PARAM_RE.findall(body)}
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "func_name": name,
            "func_args": args,
        })
    return calls


def strip_tool_markup(text):
    """Remove any leftover tool-call markup so it never reaches the teacher."""
    if not text:
        return ""
    return _TOOL_CALL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# 3. Completion loop with hybrid tools
# ---------------------------------------------------------------------------
async def get_completion_with_tools(client: OpenAI, messages: list, mcp_session: ClientSession,
                                    supabase_mcp_tools: list, stream=False):
    # token completions
    completion_tokens = 0

    # Unified tool list: local tool definitions + remote MCP tools
    combined_tools = LOCAL_TOOLS.copy()
    for mcp_tool in supabase_mcp_tools:
        combined_tools.append({
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": mcp_tool.input_schema
            }
        })

    working = messages.copy()

    # The model may need several rounds, e.g. look up a name, then fetch the profile,
    # then fetch performance data. Tools stay available on every round.
    for _ in range(MAX_TOOL_ROUNDS):
        response: ChatCompletion = client.chat.completions.create(
            model=MODEL,
            messages=working,
            tools=combined_tools,
        )
        message = response.choices[0].message
        
        # tokens used
        completion_tokens += response.usage.completion_tokens

        tools_called, tools_info = process_tool_call(response)

        # Fallback for tool calls that came back as plain text
        if not tools_called:
            tools_called = parse_text_tool_calls(message.content)
            if tools_called:
                tools_info = _assistant_tool_message(tools_called, content=None)

        # No tool calls: this is the final answer
        if not tools_called:
            final_output = strip_tool_markup(message.content) or (
                "I could not complete that request. Please try asking again."
            )
            return final_output, working + [{"role": "assistant", "content": final_output}], completion_tokens

        # Run the tools and add their results to the conversation
        working.append(tools_info)
        for tool in tools_called:
            result = await execute_tool(
                func_name=tool["func_name"],
                func_args=tool["func_args"],
                mcp_session=mcp_session,
            )
            working.append({
                "role": "tool",
                "tool_call_id": tool["id"],
                "content": tool_result_to_text(result=result),
            })

    # Too many rounds: ask for an answer from what has been retrieved so far (no tools)
    response = client.chat.completions.create(
        model=MODEL,
        messages=working + [{
            "role": "user",
            "content": "Answer now using only the information already retrieved. Do not call any more tools.",
        }],
    )
    
    # final completion tokens used
    completion_tokens += response.usage.completion_tokens
    
    final_output = strip_tool_markup(response.choices[0].message.content) or (
        "I could not finish retrieving the data. Please try a more specific request."
    )
    return final_output, working + [{"role": "assistant", "content": final_output}], completion_tokens