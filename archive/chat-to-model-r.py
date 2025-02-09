# chat-to-model.py
import streamlit as st
import os
import json
import uuid
from groq import Groq
import openturns as ot
import numpy as np
import pandas as pd

from modules.api_utils import call_groq_api_with_tools, call_groq_api_simple
from analysis_tools.sobol import perform_sobol_analysis, build_sobol_prompt

# =============================================================================
# Agent Helper: Decide Which Tool to Use
# =============================================================================
def agent_decide_tool(user_query, model_name):
    """
    Uses an LLM agent to decide if an analysis tool should be invoked.
    The available tool is "perform_sobol_analysis" which performs Sobol sensitivity analysis.
    If the query expresses interest in understanding which input is most influential (or uncertainties),
    the agent returns a JSON object with the tool name and the number of samples (if mentioned, or default to 1000).
    Otherwise, it returns an empty JSON object.
    """
    agent_prompt = f"""You are an analysis tool selection assistant.
The available analysis tool is "perform_sobol_analysis", which performs Sobol sensitivity analysis on a computational model.
It requires a parameter "sobol_samples" (the number of samples to use).
If a user query expresses interest in understanding which input is most influential and wants to quantify uncertainties,
you should select this tool. If the query includes a number, interpret that as the desired number of samples; if not, default to 1000.
If no analysis is needed, output an empty JSON object.
Return your answer as a JSON object. For example:
{{ "tool": "perform_sobol_analysis", "sobol_samples": 5000 }}
or
{{}}
User query: "{user_query}"
"""
    messages = [{"role": "system", "content": agent_prompt}]
    try:
        response = call_groq_api_simple(messages, model_name=model_name)
        agent_output = response.choices[0].message.content.strip()
        # Attempt to parse the JSON output.
        tool_decision = json.loads(agent_output)
        return tool_decision
    except Exception as e:
        # If parsing fails or an error occurs, return an empty dict.
        return {}

# =============================================================================
# Sidebar: Language Model Selection
# =============================================================================
language_model_options = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b-specdec"
]
with st.sidebar:
    selected_language_model = st.selectbox("Select Language Model", options=language_model_options)

# =============================================================================
# Main Page: Computational Model Selection & Editable Code
# =============================================================================
# Mapping from computational model names to their file paths.
computational_models = {
    "Beam": "examples/Beam.py",
    "Borehole Model": "examples/Borehole_Model.py"
}

selected_comp_model = st.selectbox("Select Computational Model", options=list(computational_models.keys()))

# When a new computational model is selected, update the text editor with its code.
if "current_comp_model" not in st.session_state:
    st.session_state.current_comp_model = selected_comp_model
    try:
        with open(computational_models[selected_comp_model], "r") as f:
            comp_model_code = f.read()
    except Exception as e:
        comp_model_code = f"Error reading file: {e}"
    st.session_state.comp_model_code = comp_model_code
elif st.session_state.current_comp_model != selected_comp_model:
    st.session_state.current_comp_model = selected_comp_model
    try:
        with open(computational_models[selected_comp_model], "r") as f:
            comp_model_code = f.read()
    except Exception as e:
        comp_model_code = f"Error reading file: {e}"
    st.session_state.comp_model_code = comp_model_code

# Display the computational model code in a text editor.
comp_model_code_editor = st.text_area(
    "Edit Computational Model Code",
    value=st.session_state.comp_model_code,
    key=selected_comp_model,
    height=300
)

# =============================================================================
# Reset Conversation Button
# =============================================================================
# When clicked, the conversation is reset and the current model code (as edited)
# is injected as a system message. (No initial greeting is provided.)
if st.button("Reset Conversation with Computational Model"):
    st.session_state.messages = [{
        "role": "system",
        "content": f"Below is the computational model code for analysis:\n\n{comp_model_code_editor}"
    }]
    st.success("Conversation reset. You can now start chatting with your model.")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": f"Below is the computational model code for analysis:\n\n{comp_model_code_editor}"
    }]

# =============================================================================
# Display Conversation (hide system and tool messages)
# =============================================================================
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        st.chat_message(msg["role"]).write(msg["content"])

# =============================================================================
# Chat Input with Agent-based Tool Selection
# =============================================================================
if prompt := st.chat_input("Type your message here..."):
    # Append the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Use the agent to decide which tool (if any) should be invoked.
    tool_decision = agent_decide_tool(prompt, model_name="llama-3.3-70b-versatile")
    
    # If the agent indicates the Sobol tool should be used:
    if tool_decision and "tool" in tool_decision and tool_decision["tool"] == "perform_sobol_analysis":
        sobol_samples = tool_decision.get("sobol_samples", 1000)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "perform_sobol_analysis",
                    "description": (
                        "Perform Sobol sensitivity analysis on the given computational model code "
                        "using a specified number of samples. Returns markdown formatted results."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sobol_samples": {
                                "type": "integer",
                                "description": "The number of samples to use for Sobol analysis."
                            },
                            "user_code": {
                                "type": "string",
                                "description": "The Python code for the computational model."
                            }
                        },
                        "required": ["sobol_samples", "user_code"]
                    }
                }
            }
        ]
        # Call the API with tool support.
        response = call_groq_api_with_tools(
            st.session_state.messages,
            model_name=selected_language_model,
            tools=tools
        )
    else:
        # No tool selected—process as a normal query.
        response = call_groq_api_simple(
            st.session_state.messages,
            model_name=selected_language_model
        )
    
    assistant_message = response.choices[0].message

    # Process tool calls if present.
    if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name == "perform_sobol_analysis":
                args = json.loads(tool_call.function.arguments)
                sobol_samples = args.get("sobol_samples")
                user_code_arg = args.get("user_code")
                # Run the local Sobol analysis function.
                sobol_results = perform_sobol_analysis(user_code_arg, sobol_samples)
                # Build the Sobol analysis interpretation prompt.
                sobol_prompt = build_sobol_prompt(sobol_results)
                # Append the tool response to the conversation with a valid tool_call_id.
                tool_response_message = {
                    "role": "tool",
                    "name": "perform_sobol_analysis",
                    "content": sobol_prompt,
                    "tool_call_id": str(uuid.uuid4())
                }
                st.session_state.messages.append(tool_response_message)
                # (Do not display tool messages to the user.)
        # After processing tool calls, call the API again (without tools)
        # so the LLM can incorporate the tool output.
        response2 = call_groq_api_simple(
            st.session_state.messages,
            model_name=selected_language_model
        )
        final_response = response2.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.chat_message("assistant").write(final_response)
    else:
        # No tool call—simply append and display the assistant's response.
        st.session_state.messages.append({"role": "assistant", "content": assistant_message.content})
        st.chat_message("assistant").write(assistant_message.content)
