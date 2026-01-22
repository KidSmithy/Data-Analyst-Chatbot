# 🤖 Integrated AI Workstation

A powerful, modular AI interface combining a **Data Analyst Agent** for code-based data analysis and a **Research Assistant Agent** for academic literature review. Built with Gradio, Google ADK, and the Model Context Protocol (MCP).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gradio](https://img.shields.io/badge/Frontend-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Key Features

### 📊 Data Analyst Agent
*   **Automated Analysis:** Upload CSV or Excel files. The agent inspects the schema, plans an analysis strategy, and writes Python code to execute it.
*   **Visualization:** Generates Matplotlib/Seaborn charts automatically and displays them in the chat.
*   **Self-Correction:** If the generated code fails, the agent enters a "Fix Mode" to debug and rewrite the script.
*   **Stateless Execution:** Runs code in a secure, local environment where variables are reset between turns to ensure reproducibility.

### 🔬 Research Assistant Agent
*   **Academic Search:** Queries **Google Scholar** and **Semantic Scholar** via MCP tools to find relevant papers.
*   **RAG (Retrieval-Augmented Generation):** Saves found papers to a local SQLite vector database (`sqlite-vec`) for semantic search in future queries.
*   **Deep Dive:** Can perform parallel "deep dives" into specific papers to extract methodologies and key findings.

> **⚠️ Note:** The Research Agent cannot be effectively run using the **SambaNova API key** (Free Tier) as it consumes massive rate limits during parallel processing. This will likely exhaust the daily rate limit before storing anything meaningful to the database.

### 🗄️ Database Manager
*   **Session History:** View, rename, or delete past chat sessions.
*   **Paper Library:** Manage your saved research papers (view abstracts, delete entries).
*   **Artifacts:** Access generated plots and converted CSV files.

---

## 🛠️ Architecture

*   **Frontend:** Gradio (`ui.py`)
*   **Orchestration:** Google Agent Development Kit (ADK) & LiteLLM
*   **Database:** SQLite with `sqlite-vec` extension for vector embeddings.
*   **Tools (MCP):**
    *   `mcp-server-data-exploration`: Handles local script execution.
    *   Smithery.ai MCPs: Connects to Scholar APIs.

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/KidSmithy/Data-Analyst-Chatbot.git
cd Data-Analyst-Chatbot
```

### 2. Install Dependencies
Ensure you have Python 3.13+ installed.
```bash
pip install -r requirements.txt
```
*Note: If `requirements.txt` is missing, install key packages:*
```bash
pip install gradio pandas matplotlib seaborn litellm python-dotenv google-genai-adk mcp sqlite-vec
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```ini
# For Data Analyst (SambaNova or OpenAI)
SAMBANOVA_API_KEY=your_sambanova_key
OPENAI_API_KEY=your_openai_key

# For Research Agent (Smithery.ai / Scholar MCPs)
SMITHERY_API_KEY=your_smithery_key
```

---

## 🖥️ Usage

### Option A: Run the Full Workstation (Default)
This launches the main interface with the **SambaNova** backend for the Data Analyst.
```bash
python main.py
```
- **URL:** `http://127.0.0.1:7860`

### Option B: Run with OpenAI Backend
If you prefer using GPT-4o (or compatible models) for the Data Analyst:
```bash
python ui_openai.py
```
- **URL:** `http://127.0.0.1:7861`

### Option C: Database Viewer Only
To manage sessions and papers without loading the agents:
```bash
python database_viewer.py
```

---

## 🧩 MCP Layer & Internals

### MCP Layer
*   **Data Exploration Server:** A MCP server (`mcp-server-data-exploration`, Source code: https://github.com/reading-plus-ai/mcp-server-data-exploration) runs as a subprocess to handle file loading and Python script execution safely.
*   **Research Tools:** Connects to external MCP servers (via Smithery) for scholar search tools.
*   **Persistence:** A SQLite database (via `database/session_manager.py`) stores chat history, sessions, and research papers.

### Server Launcher: `start_mcp.py`
This script acts as the Launcher and Adapter for the server implementation. It bridges the gap between the main chatbot application and the local MCP package.

**Operational Responsibilities:**
*   **Environment Virtualization:** Dynamically modifies `sys.path` to include the server source code, allowing the chatbot to access the logic without a standard Python package installation.
*   **Communication Protocol:** Forces Python’s standard output (`stdout`) and error (`stderr`) to be unbuffered.
    *   *Note:* This is critical for JSON-RPC communication. Buffering would cause the chatbot to hang while waiting for a response that hasn't been flushed from the system buffer.
*   **Async Execution:** Initializes the `asyncio` loop and runs the server's main entry point.

*MCP server implementation located in `src/mcp_server_ds/server.py`.*

### Capabilities
*   **`load_csv`:** Loads a CSV into a persistent Pandas DataFrame in memory.
*   **`run_script`:** Executes Python scripts against the loaded DataFrame.
*   **Security:** Prevents direct file system writes (except specific temp paths) and restricts library usage to data science essentials (Pandas, NumPy, Scipy, Matplotlib).

### Integration & Execution Flow
The chatbot does not import these files directly; instead, it manages the server as a subprocess via the `DirectMCPClient` class.

**Process Lifecycle:**
1.  **Spawn:** The `DirectMCPClient` locates `start_mcp.py` and launches it using `subprocess.Popen`.
2.  **Piping:** Standard I/O streams are connected (`stdin` for requests, `stdout` for responses).
3.  **Handshake:** The client sends an initialize request; the script sets the environment and returns the server's capabilities.
4.  **Isolation:** This architecture provides **Execution Isolation**. If a complex data script causes a crash or memory overflow, only the MCP subprocess is affected, keeping the main Chatbot UI stable.

---



## 📂 Project Structure

```text
├── agents/
│   ├── data_agent.py          # SambaNova Data Analyst logic (ADK)
│   ├── openai_data_agent.py   # OpenAI Data Analyst logic
│   ├── research_agent.py      # Research Agent (SambaNova)
│   ├── openai_research_agent.py # Research Agent (OpenAI)
│   └── agents.md              # System prompts and instructions
├── mcp-server-data-exploration/ # Local MCP Server implementation
├── database.py                # SQLite + Vector DB logic
├── database_viewer.py         # Gradio UI for DB management
├── ui.py                      # Main Chat Interface
├── ui_openai.py               # Entry point for OpenAI version
├── main.py                    # Entry point for Default version
├── config.py                  # Global settings
└── start_mcp.py               # Launcher for local MCP server
```

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 🔧 Troubleshooting

### Error: `Expecting value: line 1 column 1 (char 0)`

**Symptoms:**  
The agent fails immediately when trying to start the analysis tools, often displaying a `JSONDecodeError`.

**Cause:**  
This error means the **MCP Server subprocess crashed** silently on startup and returned no output. This is almost always caused by **missing Python packages** (e.g., `vaderSentiment`, `scikit-learn`) that were recently added to the project.

**Solution:**  
Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```