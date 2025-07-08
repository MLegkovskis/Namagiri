# **Namagiri: Conversational Sensitivity Analysis**

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)
[](https://streamlit.io)

**Namagiri** is an interactive web application that uses a Large Language Model (LLM) as a conversational interface to perform complex **Sobol Sensitivity Analysis** on computational models.

Instead of just answering questions, the LLM is equipped with a powerful "tool" – the ability to run a full uncertainty quantification analysis using the `openturns` library. Users can load a model, edit it on the fly, and simply ask the chatbot to identify the most important parameters. The system then executes the analysis, generates plots and tables, and prompts the LLM to provide a detailed, data-driven interpretation.

## ✨ Key Features

  * **Conversational Interface:** Use natural language to interact with and analyze complex computational models.
  * **LLM-Powered Tool Use:** The application intelligently decides when to call a local, high-fidelity analysis function based on the user's query.
  * **Robust Sensitivity Analysis:** Leverages the `openturns` library to perform Sobol analysis, calculating first, second, and total-order indices.
  * **Dynamic Visualizations:** Automatically generates and displays bar charts and Sobol radial plots to visualize the results.
  * **On-the-Fly Model Editing:** Load example models or write your own directly in the browser and apply them instantly to the chat session.
  * **Fast LLM Integration:** Powered by the Groq API for near-instant conversational feedback.
  * **Modular Architecture:** Analysis logic, API utilities, and prompts are cleanly separated for maintainability.

-----

## ⚙️ How It Works

The application follows a sophisticated workflow that goes beyond a simple chat-bot. When a user requests a sensitivity analysis, the system executes a series of steps to provide a rich, analytical response.

```mermaid
graph TD
    subgraph "Streamlit UI"
        A[User selects/edits a model code] --> B[User asks: 'run sobol analysis']
    end

    subgraph "Backend Logic (chat-to-model.py)"
        B --> C{Detects 'sobol' keyword}
        C --> D[Call perform_sobol_analysis()]
    end

    subgraph "Analysis Engine (analysis_tools/sobol.py)"
        D --> E[Execute model code & run OpenTURNS analysis]
        E --> F[Generate Sobol indices (S1, ST, S2)]
        F --> G[Generate Plots (Bar & Radial)]
    end

    subgraph "LLM Interpretation"
        F --> H[Format results into a detailed prompt]
        H --> I[Call Groq LLM API]
        I --> J[LLM generates natural language explanation]
    end

    subgraph "Final Output"
       G & J --> K[Display Plots & LLM response in UI]
    end
```

-----

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- |:---|
| **Web Framework** |  **Streamlit** | For creating the interactive user interface. |
| **LLM Service** |  **Groq** | Provides fast LLM inference via API. |
| **UQ & Stats** |  **OpenTURNS** | The core engine for Sobol sensitivity analysis. |
| **Data Handling** |  **NumPy** /  **Pandas** | For numerical operations and data structuring. |
| **Plotting** |  **Matplotlib** | To create the Sobol index bar and radial plots. |

-----

## 🚀 Getting Started

### \#\#\# Prerequisites

  * Python 3.9+
  * A Groq API Key

### \#\#\# Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/namagiri.git
    cd namagiri
    ```
2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Create a `requirements.txt` file** with the following content:
    ```txt
    streamlit
    groq
    openturns
    pandas
    numpy
    matplotlib
    ```
4.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
5.  **Set up your API key:**
    Create a `.env` file in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY="your-groq-api-key-here"
    ```
    The application will load this key automatically.

### \#\#\# Running the Application

Launch the Streamlit app with the following command:

```sh
streamlit run chat-to-model.py
```

Open your web browser to the local URL provided by Streamlit.

-----

## 📖 Usage

1.  **Select a Model:** Use the sidebar to choose a pre-loaded computational model (e.g., "Beam").
2.  **Edit (Optional):** Modify the model's Python code directly in the text editor.
3.  **Apply Code:** Click the **"Apply Model Code"** button. This registers the code with the chat session and displays a preview.
4.  **Start Chatting:** Ask the model general questions or trigger the analysis.
5.  **Perform Analysis:** To run the Sobol analysis, simply type a message containing the word **"sobol"** (e.g., "run sobol analysis with 2000 samples"). The application will display the generated plots and the LLM's detailed interpretation of the results.

-----

## 📂 Project Structure

```
└── ./
    ├── analysis_tools/   # Core scientific analysis modules
    │   └── sobol.py      # Functions for Sobol analysis, plotting, and prompt generation
    ├── examples/         # Sample computational models (e.g., Beam.py)
    ├── modules/          # Helper utilities for the Streamlit app
    │   ├── api_utils.py  # Wrappers for the Groq API
    │   └── system_prompt.py # Stores the base system prompt for the LLM
    └── chat-to-model.py  # The main Streamlit application file
```
