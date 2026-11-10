import json
from openai import OpenAI
from openai.types.chat import ChatCompletion
from mcp import ClientSession
from mcp.types import CallToolResult
from .data_functions import read_student_csv, read_student_csv_tool

# 1. Keep your local function mapping intact
LOCAL_FUNCTION_REGISTRY = {
    "read_student_csv": read_student_csv
}

LOCAL_TOOLS = [
    read_student_csv_tool,
]

# 2. Hybrid Function Executor (Now asynchronous to handle MCP calls)
async def execute_tool(func_name, func_args, mcp_session: ClientSession = None):
    # ROUTE A: Execute locally if it exists in your registry
    if func_name in LOCAL_FUNCTION_REGISTRY:
        results = LOCAL_FUNCTION_REGISTRY[func_name](**func_args)
        return results
    
    # ROUTE B: Route over the network to Supabase if an active MCP session exists
    elif mcp_session is not None:
        try:
            mcp_result = await mcp_session.call_tool(name=func_name, arguments=func_args)
            # The MCP spec wraps structural text data inside content objects
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

    # MCP content list (what you have now: result = response.content)
    if isinstance(result, list) and any(hasattr(b, "type") and hasattr(b, "model_dump") for b in result):
        return "\n".join(_block_to_text(b) for b in result)

    # Local tool results
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except TypeError:
        return str(result)


# Tool calls extraction (Unchanged, completely compatible)
def process_tool_call(response: ChatCompletion):
    tools_called = []
    message = response.choices[0].message
    
    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return None, None
    
    for tool in message.tool_calls:
        tools_called.append({
            "id": tool.id,
            "func_name": tool.function.name,
            "func_args": json.loads(tool.function.arguments),
        })

    tools_info = {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [{
            "id": tool["id"],
            "type": "function",
            "function": {
                "name": tool["func_name"],
                "arguments": json.dumps(tool["func_args"]),
            }
        } for tool in tools_called ]
    }
    
    return tools_called, tools_info

# 3. Completion loop with hybrid tools (Updated to async def)
async def get_completion_with_tools(client: OpenAI, messages: list, mcp_session: ClientSession, supabase_mcp_tools: list, stream=False):
    
    # Compile unified flat list: Local Tool Definitions + Formatted Remote MCP Tools
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

    # Get first completion with combined tools schema list
    response: ChatCompletion = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=messages,
        tools=combined_tools,
        stream=stream
    )
    
    message = response.choices[0].message
    
    # check for tool calls in response
    tools_called, tools_info = process_tool_call(response=response)
    
    if not tools_called:
        return message.content, messages + [{
            "role": "assistant", "content": message.content
        }]
    
    # Asynchronous Execution Loop
    tools_executed = []
    for tool in tools_called:
        # Passes execution tasks to either local runtime or the MCP network tunnel
        result = await execute_tool(
            func_name=tool["func_name"], 
            func_args=tool["func_args"], 
            mcp_session=mcp_session
        )
        
        tools_executed.append({
            "role": "tool",
            "tool_call_id": tool["id"],
            "content": tool_result_to_text(result=result)
        })
        
    #* update the messages (shallow copy of current messages), append tools info and extend execution info.
    updated_messages: list = messages.copy()
    
    updated_messages.append(tools_info)
    updated_messages.extend(tools_executed)
    
    #! get final response with processed tool calls and their results.
    
    final_completion: ChatCompletion = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=updated_messages,
        stream=False
    )
    
    final_output = final_completion.choices[0].message.content
        
            
    return final_output, updated_messages + [{
        "role": "assistant",
        "content": final_output
    }]