from typing import List, Dict, Any, Optional
from openai import OpenAI
import re
from urllib.parse import urlparse
import time

def extract_url_root_domain(url):
    """
    Extract root domain from URL
    Examples:
    - https://www.example.com/path -> example.com
    - sub.example.co.uk -> example.co.uk
    """
    # Ensure URL contains protocol, add if missing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Use urlparse to parse URL
    parsed = urlparse(url).netloc
    if not parsed:
        parsed = url
        
    # Remove port number (if exists)
    parsed = parsed.split(':')[0]
    
    # Split domain parts
    parts = parsed.split('.')
    
    # Handle special second-level domains like .co.uk, .com.cn etc.
    if len(parts) > 2:
        if parts[-2] in ['co', 'com', 'org', 'gov', 'edu', 'net']:
            if parts[-1] in ['uk', 'cn', 'jp', 'br', 'in']:
                return '.'.join(parts[-3:])
    
    # Return main domain part (last two parts)
    return '.'.join(parts[-2:])

def get_clean_content(line):
    clean_line = re.sub(r'^[\*\-•#\d\.]+\s*', '', line).strip()
    clean_line = re.sub(r'^[\'"]|[\'"]$', '', clean_line).strip()
    if (clean_line.startswith('"') and clean_line.endswith('"')) or \
    (clean_line.startswith("'") and clean_line.endswith("'")):
        clean_line = clean_line[1:-1]
    return clean_line

def get_content_from_tag(content, tag, default_value=None):
    # Notes:
    # 1) (.*?) lazy match, match as few characters as possible
    # 2) (?=(</tag>|<\w+|$)) lookahead, stops when followed by </tag> or <tag starting with word char> or end of text
    # 3) re.DOTALL makes dot . match newline characters
    pattern = rf"<{tag}>(.*?)(?=(</{tag}>|<\w+|$))"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return default_value


def get_response_from_llm(
        messages: List[Dict[str, Any]],
        client: OpenAI,
        model: str,
        stream: Optional[bool] = False,
        temperature: Optional[float] = 0.6,
        depth: int = 0
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream
        )
        if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
            content = response.choices[0].message.content
        return {
            "content": content.strip()
        }
    except Exception as e:
        print(f"LLM API error: {e}")
        if "Input data may contain inappropriate content" in str(e):
            return {
                "content": ""
            }
        if "Error code: 400" in str(e):
            return {
                "content": ""
            }
        if depth < 512:
            time.sleep(1)
            return get_response_from_llm(messages=messages, client=client, model=model, stream=stream, temperature=temperature, depth=depth+1)
        raise e
