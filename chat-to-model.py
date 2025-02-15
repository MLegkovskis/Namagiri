import streamlit as st
import os
import json
import uuid
from groq import Groq
import openturns as ot
import numpy as np
import pandas as pd
from modules.system_prompt import SYSTEM_PROMPT
from modules.api_utils import call_groq_api_with_tools, call_groq_api_simple
from analysis_tools.sobol import (
    perform_sobol_analysis, 
    build_sobol_prompt,
    plot_sobol_indices
)

# =============================================================================
# Initialize Session State Defaults
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "registered_model_code" not in st.session_state:
    st.session_state.registered_model_code = ""

if "current_comp_model" not in st.session_state:
    st.session_state.current_comp_model = None

if "comp_model_code" not in st.session_state:
    st.session_state.comp_model_code = ""

# =============================================================================
# Sidebar: Model Settings, Code Editor, and Apply Button
# =============================================================================
with st.sidebar:
    st.header("Model Settings")
    
    # Language Model Selection
    language_model_options = ["llama-3.3-70b-versatile"]
    selected_language_model = st.selectbox("Select Language Model", options=language_model_options)
    
    # Computational Model Selection
    computational_models = {
        "Beam": "examples/Beam.py",
        "Borehole Model": "examples/Borehole_Model.py"
    }
    selected_comp_model = st.selectbox("Select Computational Model", options=list(computational_models.keys()))
    
    # When the computational model selection changes, update the code.
    if st.session_state.current_comp_model != selected_comp_model:
        st.session_state.current_comp_model = selected_comp_model
        try:
            with open(computational_models[selected_comp_model], "r") as f:
                comp_model_code = f.read()
        except Exception as e:
            comp_model_code = f"# Error reading file: {e}"
        st.session_state.comp_model_code = comp_model_code
        # Optionally, update the registered code immediately:
        st.session_state.registered_model_code = comp_model_code

    st.subheader("Edit Computational Model Code")
    comp_model_code_editor = st.text_area(
        label="",
        value=st.session_state.comp_model_code,
        key=f"code_editor_{selected_comp_model}",
        height=300
    )
    
    # Apply button to register the current code.
    if st.button("Apply Model Code"):
        try:
            # Execute code in a fresh namespace to verify that it defines both 'model' and 'problem'
            local_namespace = {}
            exec(comp_model_code_editor, local_namespace)
            if 'model' not in local_namespace or 'problem' not in local_namespace:
                st.error("Error: The model code must define both 'model' and 'problem'.")
            else:
                st.session_state.registered_model_code = comp_model_code_editor
                st.session_state.comp_model_code = comp_model_code_editor
                # Update the system message with the registered model code.
                system_message = {
                    "role": "system",
                    "content": f"{SYSTEM_PROMPT}\n\n**Registered Model Code:**\n```python\n{st.session_state.registered_model_code}\n```"
                }

                if st.session_state.messages:
                    if st.session_state.messages[0]["role"] == "system":
                        st.session_state.messages[0] = system_message
                    else:
                        st.session_state.messages.insert(0, system_message)
                else:
                    st.session_state.messages.append(system_message)
                st.success("Model code applied successfully.")
        except Exception as e:
            st.error(f"Error applying model code: {e}")

# =============================================================================
# Main Page: Code Preview and Chat Interface
# =============================================================================
st.title("Computational Model Chat Interface")

# Display a syntax-highlighted preview of the registered code.
if st.session_state.registered_model_code.strip():
    st.subheader("Registered Computational Model Code Preview")
    st.code(st.session_state.registered_model_code, language="python")
else:
    st.info("No computational model code is registered. Please apply your model code from the sidebar.")

# Display conversation messages.
# Only render messages with role "user" or "assistant" (system and tool messages are hidden).
for msg in st.session_state.messages:
    if msg["role"] not in ["user", "assistant"]:
        continue
    # Also skip any assistant messages that are raw tool call representations.
    if msg["role"] == "assistant" and msg["content"].strip().startswith("<function="):
        continue
    st.chat_message(msg["role"]).write(msg["content"])

# =============================================================================
# Chat Input and Message Processing (including Sobol Analysis Tool Calls)
# =============================================================================
if st.session_state.registered_model_code.strip():
    prompt = st.chat_input("Type your message here...")
    if prompt:
        # Append the user's message.
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Decide whether to include the Sobol tool based on the user's message.
        if "sobol" in prompt.lower():
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
            response = call_groq_api_with_tools(
                st.session_state.messages,
                model_name=selected_language_model,
                tools=tools
            )
        else:
            response = call_groq_api_simple(
                st.session_state.messages,
                model_name=selected_language_model
            )
        
        assistant_message = response.choices[0].message
        
        # Process any tool calls if they exist.
        if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "perform_sobol_analysis":
                    # Always use the latest registered model code from session state.
                    user_code_arg = st.session_state.registered_model_code
                    args = json.loads(tool_call.function.arguments)
                    sobol_samples = args.get("sobol_samples")
                    if not sobol_samples:
                        st.error("Please specify the number of samples for the Sobol analysis.")
                        continue
                    
                    try:
                        sobol_results = perform_sobol_analysis(user_code_arg, sobol_samples)
                    except Exception as e:
                        st.error(f"Error during Sobol analysis: {e}")
                        continue

                    # Build a prompt for the LLM using the Sobol results.
                    sobol_prompt = build_sobol_prompt(sobol_results)
                    tool_response_message = {
                        "role": "tool",
                        "name": "perform_sobol_analysis",
                        "content": sobol_prompt,
                        "tool_call_id": str(uuid.uuid4())
                    }
                    st.session_state.messages.append(tool_response_message)
                    
                    # Display the Sobol analysis plot.
                    fig = plot_sobol_indices(sobol_results)
                    if fig is not None:
                        st.pyplot(fig)
            
            # After processing tool calls, request a final assistant response.
            response2 = call_groq_api_simple(st.session_state.messages, model_name=selected_language_model)
            final_response = response2.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            st.chat_message("assistant").write(final_response)
        else:
            # If no tool calls, simply append the assistant's reply.
            if not assistant_message.content.strip().startswith("<function="):
                st.session_state.messages.append({"role": "assistant", "content": assistant_message.content})
                st.chat_message("assistant").write(assistant_message.content)
else:
    st.info("Please apply your computational model code from the sidebar.")
