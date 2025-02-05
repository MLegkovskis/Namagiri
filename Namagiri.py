import os
import streamlit as st
import openturns as ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from groq import Groq
import openai

# Import our new Sobol modules and context prompt module
from modules.Sobol_Method.sobol import perform_sobol_analysis
from modules.Sobol_Method.sobol_prompt import build_sobol_prompt
from modules.context_prompt import build_context_prompt

# =============================================================================
# LLM API Call via Groq
# =============================================================================
def call_groq_api(prompt, model_name="gemma2-9b-it"):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=model_name
    )
    response_text = chat_completion.choices[0].message.content
    return response_text

# =============================================================================
# Streamlit Page Setup and Sidebar (Model Selection & Code Editor)
# =============================================================================
st.set_page_config(page_title="Chat with Your Computational Model", layout="wide")
st.title("Chat with Your Computational Model")

# Path to the examples directory (assumed same as your original app)
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'examples')

def get_available_models(directory):
    try:
        files = os.listdir(directory)
        model_files = [f for f in files if f.endswith('.py') and not f.startswith('__')]
        model_names = [os.path.splitext(f)[0] for f in model_files]
        return model_names, model_files
    except FileNotFoundError:
        st.error(f"The directory '{directory}' does not exist.")
        return [], []

model_names, model_files = get_available_models(EXAMPLES_DIR)
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a model",
    options=model_names,
    index=0 if model_names else None
)

def load_model_code(model_filename):
    model_path = os.path.join(EXAMPLES_DIR, model_filename)
    try:
        with open(model_path, 'r') as file:
            code = file.read()
        return code
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return ""

if selected_model_name:
    selected_model_file = [f for f in model_files if os.path.splitext(f)[0] == selected_model_name][0]
    default_code = load_model_code(selected_model_file)
else:
    default_code = "# Select a model from the sidebar to display its code here."

user_code = st.text_area("Model Code", value=default_code, height=400)
sobol_samples = st.number_input("Number of Sobol samples", min_value=100, max_value=1000000, value=1000, step=100)

# =============================================================================
# Chat Interface (Session State for conversation and module contexts)
# =============================================================================
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "module_context" not in st.session_state:
    st.session_state.module_context = {}  # Dictionary to hold context from various modules

chat_input = st.text_input("Enter your message:", key="chat_input")

def add_message(role, content):
    st.session_state.conversation.append({"role": role, "content": content})

# Display the conversation
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Agent:** {msg['content']}")

# =============================================================================
# Process User Input and Decide on Action (Module-specific analysis or generic query)
# =============================================================================
if st.button("Send"):
    if chat_input:
        user_message = chat_input
        add_message("user", user_message)
        
        # If the user's message suggests interest in Sobol analysis, run that module
        if "most important" in user_message.lower() or "sobol" in user_message.lower():
            try:
                sobol_results = perform_sobol_analysis(user_code, int(sobol_samples))
                st.session_state.module_context["Sobol"] = sobol_results  # Save context for the Sobol module
                
                # Build the LLM prompt using the Sobol prompt builder
                prompt = build_sobol_prompt(sobol_results)
            except Exception as e:
                error_msg = f"Error performing Sobol analysis:\n{traceback.format_exc()}"
                add_message("agent", error_msg)
                st.rerun()
            
            llm_response = call_groq_api(prompt)
            add_message("agent", llm_response)
        else:
            # For generic queries, aggregate context from available modules
            prompt = build_context_prompt(user_message, st.session_state.module_context)
            llm_response = call_groq_api(prompt)
            add_message("agent", llm_response)
            
        st.rerun()
