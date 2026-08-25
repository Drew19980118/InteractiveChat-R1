"""
IGPO Tool Server - System Prompt Initialization

Uses Jinja2 templates to generate prompts with tool definitions.
Compatible with the original tools_server_ant version.
"""

import os
from time import strftime, gmtime
from typing import Dict, Any, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from jinja2 import Template, StrictUndefined
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False

from tools_server.tools import get_tools


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    """
    Render a Jinja2 template with the given variables.
    """
    if not HAS_JINJA:
        # Fallback: simple string replacement (limited functionality)
        result = template
        result = result.replace("{{today}}", str(variables.get('today', '')))
        return result
    
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


def initialize_system_prompt(
    system_prompt_template: str,
    tools: Dict[str, Dict],
    today: str
) -> str:
    """
    Initialize system prompt with tool definitions.
    """
    return populate_template(
        system_prompt_template,
        variables={
            "tools": tools,
            "today": today
        },
    )


def load_system_prompt(config_path: Optional[str] = None) -> str:
    """
    Load and initialize system prompt from config.
    """
    today = strftime("%Y-%m-%d", gmtime())
    
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    # Load config
    config = {}
    if HAS_YAML and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[tools_server] Error loading config: {e}")
    
    # Get tools
    tools = get_tools(config)
    
    # Get template
    system_prompt_template = config.get('system_prompt', '')
    
    if system_prompt_template:
        return initialize_system_prompt(system_prompt_template, tools, today)
    
    # Fallback to default prompt
    return _get_default_prompt(tools, today)


def _get_default_prompt(tools: Dict[str, Dict], today: str) -> str:
    """Get default system prompt if config is missing."""
    tool_list = "\n".join([
        f"- {name}: {tool['description']}"
        for name, tool in tools.items()
    ])
    
    tool_signatures = "\n".join([
        f'{{"type": "function", "function": {{"name": "{tool["name"]}", "description": "{tool["description"]}", "parameters": {{"type": "object", "properties": {tool["inputs"]}, "example": {tool["example"]}}}}}}}'
        for tool in tools.values()
    ])
    
    return f"""## Background information 
* Today is {today}
* You are Deep AI Research Assistant

The question I give you is a complex question that requires a *deep research* to answer.
I will provide you with tools to help you answer the question:
{tool_list}

You don't have to answer the question now, but you should first think about the research plan or what to search next.
Your output format should be exactly one of the following two formats:

<think>
YOUR THINKING PROCESS
</think>
<answer>
YOUR ANSWER AFTER GETTING ENOUGH INFORMATION
</answer>

or

<think>
YOUR THINKING PROCESS
</think>
<tool_call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool_call>

You should always follow one of the above two formats strictly. Only output a
final answer (in words, numbers or phrase) inside the <answer></answer> tag,
without any explanations or extra information. If this is a yes-or-no question,
you should only answer yes or no.


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_signatures}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""


# Initialize SYSTEM_PROMPT at module import
try:
    SYSTEM_PROMPT = load_system_prompt()
    print("[tools_server] System prompt initialized successfully")
except Exception as e:
    print(f"[tools_server] Error initializing system prompt: {e}")
    SYSTEM_PROMPT = _get_default_prompt(
        {"web_search": {"name": "web_search", "description": "Search the web", "inputs": {}, "example": {}}},
        strftime("%Y-%m-%d", gmtime())
    )


if __name__ == "__main__":
    print("=" * 60)
    print("SYSTEM_PROMPT:")
    print("=" * 60)
    print(SYSTEM_PROMPT)
