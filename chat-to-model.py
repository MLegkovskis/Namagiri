import streamlit as st
import os
import json
import uuid
from groq import Groq
import openturns as ot
import numpy as np
import pandas as pd

from modules.api_utils import call_groq_api_with_tools, call_groq_api_simple
from analysis_tools.sobol import (
    perform_sobol_analysis, 
    build_sobol_prompt,
    plot_sobol_indices
)

# =============================================================================
# Sidebar: Language Model Selection
# =============================================================================
language_model_options = [
    "llama-3.3-70b-versatile"
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
if "current_comp_model" not in st.session_state or st.session_state.current_comp_model != selected_comp_model:
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
    key=f"code_editor_{selected_comp_model}",
    height=300
)

# =============================================================================
# Model Registration
# =============================================================================
# Add a button that "registers" the model code by executing it and resetting the conversation.
if st.button("Register Model Code"):
    try:
        # Use a fresh namespace to execute the model code.
        local_namespace = {}
        exec(comp_model_code_editor, local_namespace)
        if 'model' not in local_namespace or 'problem' not in local_namespace:
            st.error("Error: The model code must define both 'model' and 'problem'.")
        else:
            st.session_state.registered_model_code = comp_model_code_editor
            # Reset the conversation with a system message that includes the registered code.
            st.session_state.messages = [{
                "role": "system",
                "content": f"Below is the registered computational model code for analysis:\n\n{comp_model_code_editor}"
            }]
            st.success("Model code registered successfully. You can now start chatting with your model.")
    except Exception as e:
        st.error(f"Error registering model code: {e}")

# If no model has been registered yet, warn the user.
if "registered_model_code" not in st.session_state:
    st.info("Please register your model code using the 'Register Model Code' button before interacting with the chatbot.")

# =============================================================================
# Display Conversation (hide system and tool messages)
# =============================================================================
if "messages" not in st.session_state:
    # If there is no conversation yet, initialize with the registered model code
    if "registered_model_code" in st.session_state:
        st.session_state.messages = [{
            "role": "system",
            "content": f"Below is the registered computational model code for analysis:\n\n{st.session_state.registered_model_code}"
        }]
    else:
        st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        st.chat_message(msg["role"]).write(msg["content"])

# =============================================================================
# Chat Input with Optional Tool Use for Sobol Analysis
# =============================================================================
if st.session_state.get("registered_model_code"):
    if prompt := st.chat_input("Type your message here..."):
        # Append the user's message.
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Decide whether to include the Sobol tool based on the prompt.
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
            # Call the API with tool support.
            response = call_groq_api_with_tools(st.session_state.messages, model_name=selected_language_model, tools=tools)
        else:
            # For a normal query, do not include any tools.
            response = call_groq_api_simple(st.session_state.messages, model_name=selected_language_model)
        
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
                    # Do not display the raw tool message to the user.
                    # Instead, generate and display the plot.
                    fig = plot_sobol_indices(sobol_results)
                    if fig is not None:
                        st.pyplot(fig)
            # After processing tool calls, call the API again (without tools)
            # to let the LLM use the tool output.
            response2 = call_groq_api_simple(st.session_state.messages, model_name=selected_language_model)
            final_response = response2.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            st.chat_message("assistant").write(final_response)
        else:
            # No tool call—simply append and display the assistant's response.
            st.session_state.messages.append({"role": "assistant", "content": assistant_message.content})
            st.chat_message("assistant").write(assistant_message.content)
else:
    st.info("Please register your model code first.")
