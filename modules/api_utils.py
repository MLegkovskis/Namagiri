# modules/api_utils.py
import os
from groq import Groq

def call_groq_api_with_tools(messages, model_name, tools):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_completion_tokens=4096
    )
    return response

def call_groq_api_simple(messages, model_name):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=4096
    )
    return response
