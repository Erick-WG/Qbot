# this file contains all interactions with ai.
import json
from openai import OpenAI
from openai.types.chat import ChatCompletion
from .data_functions import read_student_csv, read_student_csv_tool



FUNCTION_REGISTRY = {
    "read_student_csv": read_student_csv
}

TOOLS = [
    read_student_csv_tool,
]



# function executor
def execute_tool(func_name, func_args):
    if func_name in FUNCTION_REGISTRY:
        results = FUNCTION_REGISTRY[func_name](**func_args)
        return results
    else:
        return {"status": "error", "content": f"No function called {func_name} found!"}



# tool calls extraction
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


# completion with tool calls
def get_completion_with_tools(client: OpenAI, messages: list, stream=False):
    # get first completion.
    response: ChatCompletion = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=messages,
        tools=TOOLS,
        stream=stream
    )
    
    message = response.choices[0].message
    
    # check for tool calls in response
    tools_called, tools_info = process_tool_call(response=response)
    
    if not tools_called:
        return message.content, messages + [{
            "role": "assistant", "content": message.content
        }]
    
    # if we have tools, we need to execute the functions and get a final output.
    # tool execution.
    #! tools execution info to be in updated messages.
    tools_executed = []
    for tool in tools_called:
        result = execute_tool(func_name=tool["func_name"], func_args=tool["func_args"])
        tools_executed.append({
            "role": "tool",
            "tool_call_id": tool["id"],
            "content": json.dumps(result)
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