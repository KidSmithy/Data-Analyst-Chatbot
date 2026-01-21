
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import asyncio
import time
import base64
import uuid
import json
import subprocess
import tempfile
import traceback
import contextlib
import warnings
import litellm
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Important for backend
import matplotlib.pyplot as plt
from datetime import datetime
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Internal Imports
from config import GLOBAL_MEMORY, MODEL_NAME
from database import session_manager
from utils import convert_excel_to_csv
from pathlib import Path

# Configure LiteLLM
litellm.drop_params = True

# Suppress Pydantic warnings from LiteLLM
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

class SambaNovaLiteLlm(LiteLlm):
    """
    Custom LiteLlm wrapper for SambaNova that forces message content to be a string.
    The default ADK LiteLlm sends content as a list of parts, which gpt-oss-120b rejects.
    """
    async def generate_content(self, contents, tools=None, **kwargs):
        # 1. Convert ADK contents to LiteLLM messages (forcing string content)
        messages = []
        
        # Track tool call IDs to map responses to calls (Simple FIFO per function name)
        pending_tool_calls = {} 

        for c in contents:
            role = c.role
            if role == "model": role = "assistant"
            
            # Flatten text parts to a single string
            text_parts = [p.text for p in c.parts if p.text]
            content = " ".join(text_parts) if text_parts else ""
            
            msg = {"role": role, "content": content}
            
            # Handle Tool Calls (if any)
            tool_calls = []
            for p in c.parts:
                if p.function_call:
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    fc = p.function_call
                    
                    # Store ID for matching response later
                    if fc.name not in pending_tool_calls:
                        pending_tool_calls[fc.name] = []
                    pending_tool_calls[fc.name].append(call_id)

                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(fc.args)
                        }
                    })
            
            if tool_calls:
                msg["tool_calls"] = tool_calls
            
            # Handle Function Responses (Tool Outputs)
            for p in c.parts:
                if p.function_response:
                    msg["role"] = "tool"
                    fr = p.function_response
                    msg["name"] = fr.name
                    msg["content"] = json.dumps(fr.response) # Force string content
                    
                    # Retrieve matching ID
                    if fr.name in pending_tool_calls and pending_tool_calls[fr.name]:
                        msg["tool_call_id"] = pending_tool_calls[fr.name].pop(0)
                    else:
                        msg["tool_call_id"] = "unknown_call_id"

            messages.append(msg)
            
        # 2. Handle tools
        llm_tools = None
        if tools:
            llm_tools = []
            for tool in tools:
                try:
                    # Use litellm's utility to convert function to tool definition
                    tool_def = litellm.utils.function_to_dict(tool)
                    llm_tools.append(tool_def)
                except Exception as e:
                    print(f"⚠️ Tool conversion failed for {tool}: {e}")

        # 3. Call LiteLLM directly
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            tools=llm_tools,
            **kwargs
        )
        
        # 4. Convert LiteLLM response back to ADK format
        choice = response.choices[0]
        message = choice.message
        
        parts = []
        if message.content:
            parts.append(types.Part(text=message.content))
        
        if message.tool_calls:
            for tc in message.tool_calls:
                parts.append(types.Part(function_call=types.FunctionCall(name=tc.function.name, args=json.loads(tc.function.arguments))))
                
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=choice.finish_reason, index=0)])

# --- Direct MCP Client (Restored) ---
class DirectMCPClient:
    def __init__(self):
        self.process = None
        self.request_id = 0
        
    def start_server(self):
        print("🚀 [System] Starting MCP server subprocess...")
        # Calculate path to start_mcp.py relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        script_path = os.path.join(root_dir, "start_mcp.py")
        
        self.process = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8"
        )
        time.sleep(2)
        self._initialize()

    def _initialize(self):
        print("🔌 [MCP] Sending Initialize Request...")
        self._send_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "client", "version": "1.0"}}})
        print("✅ [MCP] Initialized successfully")

    def _send_request(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def call_tool(self, tool_name, arguments):
        print(f"🛠️ [MCP] Calling tool '{tool_name}' with args: {str(arguments)[:500]}...", file=sys.stderr)
        
        # Parameter mapping
        if tool_name == "load_csv":
            if "path" in arguments:
                arguments = {"csv_path": arguments["path"]}
            elif "csv_path" in arguments:
                pass
        elif tool_name == "run_script":
            if "script" in arguments:
                pass

        res = self._send_request({
            "jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}
        })
        
        if "error" in res:
            print(f"❌ [MCP Error] {res['error']['message']}")
            return f"Error: {res['error']['message']}"
            
        content = res["result"].get("content", [])
        text_content = [c["text"] for c in content if c["type"] == "text"]
        result_str = "\n".join(text_content) if text_content else str(res["result"])
        return result_str

    def _next_id(self):
        self.request_id += 1
        return self.request_id

    def close(self):
        """Terminates the MCP server subprocess."""
        if self.process:
            print("🛑 [System] Stopping MCP server...")
            self.process.terminate()
            self.process.wait()
            print("✅ [System] MCP server stopped")

async def run_analysis_pipeline(client, task_description, previous_error=None, previous_code=None):
    """
    Orchestrates the analysis using Google ADK Agents.
    - Phase 0: Inspector (if schema missing)
    - Initial Run: Uses SequentialAgent [Planner -> Coder]
    - Fix Mode: Uses single LlmAgent [Fixer]
    """
    csv_path = GLOBAL_MEMORY.get("current_csv", "data.csv")
    schema_info = GLOBAL_MEMORY.get("schema_info")
    
    # Shared System Prompt for Coder/Fixer
    coder_system_prompt = (
        f"You are an expert Data Analysis & Visualization Assistant. Write code to analyze this CSV file: {csv_path}\n"
        f"The dataframe is ALREADY LOADED as variable 'df'. Do NOT load it again.\n\n"
        f"=== DATASET CONTEXT ===\n{schema_info}\n\n"
        "=== CORE PRINCIPLES ===\n"
        "1. NEVER drop data without explicit user request or clear justification\n"
        "2. Preserve all original data unless cleaning is absolutely necessary\n"
        "3. Always document what changes you make to the data\n"
        "4. Your outputs must match professional data analysis standards\n\n"
        
        "=== CODE REQUIREMENTS ===\n"
        "1. Import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np, base64, io\n"
        "2. FIRST LINE MUST BE: plt.switch_backend('Agg')\n"
        "3. Set style: sns.set_theme(style='whitegrid'), plt.rcParams['figure.figsize'] = (10, 6)\n"
        "4. Use consistent color palette: 'viridis' for sequential, 'Set3' for categorical\n"
        "5. SAFETY: Do NOT convert columns with NaNs to integer (astype(int)). Use 'Int64' or fillna() first.\n"
        "6. SAFETY: Ensure all variables (especially mapping dictionaries like 'abbr_to_name') are explicitly defined in the script before use.\n"
        "7. CRITICAL: STATELESS EXECUTION. The environment is reset for every script. You must CODE FROM SCRATCH.\n"
        "8. DO NOT assume variables from previous turns exist. Only 'df' exists initially.\n"
        "9. If you need a derived dataframe (e.g., 'df_ana') or a function, you MUST define it in the current script.\n"
        "10. NEVER use a variable unless you have assigned it a value in THIS script.\n\n"
        
        "=== DATA PROCESSING PIPELINE ===\n"
        "1. DATA ACCESS: Work with the existing variable 'df'.\n"
        "2. INSPECT: Print shape, columns, dtypes, and missing values table\n"
        "3. CLEAN - BE CONSERVATIVE:\n"
        "   - DO NOT drop columns automatically - even IDs might be needed\n"
        "   - For missing values: FIRST analyze pattern, then decide:\n"
        "     * If <1% missing AND random: consider dropping rows\n"
        "     * Otherwise: fill with appropriate strategy OR leave as-is\n"
        "     * ALWAYS report missing values before taking any action\n"
        "     * NEVER use astype(int) on columns with NaNs. It will crash.\n"
        "   - Remove ONLY exact duplicate rows (df.duplicated().sum())\n"
        "   - Create derived columns only when they add clear value\n"
        "4. CREATE ANALYSIS COPY: work_df = df.copy() for any transformations\n\n"
        
        "=== CRITICAL: DATA PRESERVATION RULES ===\n"
        "1. NEVER drop columns based on name patterns (ID, UUID, etc.)\n"
        "2. If a column has many unique values, it might be categorical with high cardinality - handle appropriately\n"
        "3. Before any data modification, print: 'DATA CLEANING REPORT:'\n"
        "4. Document every change: 'Changed X: reason Y, impact Z'\n"
        "5. If unsure about dropping, KEEP THE DATA and analyze as-is\n\n"
        
        "=== ANALYSIS STRATEGY ===\n"
        "IF USER REQUEST IS SPECIFIC: Execute exactly\n"
        "IF REQUEST IS GENERAL (e.g., 'analyze', 'explore', 'show insights'):\n"
        "   --- PHASE 1: DATA OVERVIEW ---\n"
        "   1. Print '=== Data Overview ===' header\n"
        "   2. Print shape: 'Shape: (rows, columns)'\n"
        "   3. Print columns with data types\n"
        "   4. Show first 5-10 rows\n"
        "   5. Calculate and show missing values percentage for each column\n"
        "   6. Calculate and show duplicate rows count\n"
        "   7. For each column, show basic statistics based on type\n\n"
        
        "   --- PHASE 2: COLUMN-SPECIFIC ANALYSIS ---\n"
        "   FOR EACH COLUMN (in order of data importance):\n"
        "   1. Print column name and type\n"
        "   2. If categorical/object:\n"
        "      - Calculate value_counts() (show top 20 if many categories)\n"
        "      - Calculate unique count and cardinality\n"
        "      - Create horizontal bar chart for top categories\n"
        "   3. If numerical:\n"
        "      - Calculate full statistics\n"
        "      - Check for outliers using IQR method\n"
        "      - Create distribution plot (histogram + boxplot)\n"
        "   4. If datetime:\n"
        "      - Parse properly, show range\n"
        "      - Create time series analysis\n\n"
        
        "   --- PHASE 3: RELATIONSHIP ANALYSIS ---\n"
        "   1. Identify potential relationships between columns\n"
        "   2. Create cross-tabulations for categorical pairs\n"
        "   3. Calculate correlations for numerical pairs\n"
        "   4. Create grouped analyses (like 'Top 10 X per Category Y')\n"
        "   5. For hierarchical data, create nested/grouped visualizations\n\n"
        
        "   --- PHASE 4: INSIGHTS GENERATION ---\n"
        "   1. Print '=== Key Insights ===' header\n"
        "   2. List 5-10 most important findings\n"
        "   3. For each insight, include:\n"
        "      - What was found\n"
        "      - Why it matters\n"
        "      - Supporting statistics\n"
        "   4. End with '=== Recommendations ===' for further analysis\n\n"
        
        "=== VISUALIZATION STANDARDS ===\n"
        "1. Every chart must be self-explanatory\n"
        "2. Use appropriate chart types:\n"
        "   - Bar charts for categorical comparisons\n"
        "   - Histograms for distributions\n"
        "   - Scatter plots for relationships\n"
        "   - Line charts for trends\n"
        "3. Save plots using:\n"
        "3. CRITICAL: The report generator CANNOT see the images. You MUST print a text summary of the data shown in the plot immediately before saving it.\n"
        "   - Example: print(f'Plotting distribution. Mean: 10.5, Std: 2.1')\n"
        "   - Example: print(df_grouped.head().to_string())\n"
        "4. Save plots using:\n"
        "   buf = io.BytesIO()\n"
        "   plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')\n"
        "   buf.seek(0)\n"
        "   img_str = base64.b64encode(buf.read()).decode()\n"
        "   print('IMAGE_BASE64:' + img_str)\n"
        "   plt.clf()\n\n"
        
        "=== DATA ETHICS & BEST PRACTICES ===\n"
        "1. PRESERVE data integrity above all\n"
        "2. If data quality issues exist, REPORT them, don't silently fix\n"
        "3. Document all assumptions made during analysis\n"
        "4. Highlight limitations of the analysis\n"
        "5. Never make irreversible changes to data\n\n"
        
        "=== OUTPUT FORMAT ===\n"
        "1. Use clear print statements with section headers\n"
        "2. Structure output logically\n"
        "3. Include both raw numbers and interpretations\n"
        "4. Output ONLY valid Python code, no markdown\n"
        "5. Make code readable with comments\n"
        "6. Do NOT wrap your code in try-except blocks that hide errors. Let exceptions raise so the system can catch them.\n"
    )

    if previous_error:
        # --- FIX MODE: Single Agent ---
        print(f"🔧 [Coder Agent] Entering FIX mode...")
        user_content = (
            f"I tried to run your previous code, but it failed.\n\n"
            f"YOUR PREVIOUS CODE:\n{previous_code}\n\n"
            f"THE ERROR MESSAGE:\n{previous_error}\n\n"
            f"Please fix the code to resolve this error. Return the FULL corrected script."
        )
        
        try:
            messages = [
                {"role": "system", "content": coder_system_prompt},
                {"role": "user", "content": user_content}
            ]
            response = await litellm.acompletion(model=f"sambanova/{MODEL_NAME}", messages=messages)
            text = response.choices[0].message.content
            print(f"DEBUG: [Fixer Raw Output]\n{text}\n-------------------", file=sys.stderr)
            return text.replace("```python", "").replace("```", "").strip()
        except Exception as e:
            print(f"❌ Fixer Error: {e}")
            return ""
            
    else:
        # --- INITIAL MODE: Sequential Agent (Planner -> Coder) ---
        print(f"🚀 [Sequential Agent] Starting Plan -> Code pipeline...")
        agents = []
        
        # 1. Inspector Agent (Add to sequence ONLY if schema is missing)
        if not schema_info or schema_info.startswith("Unknown"):
            print("   + Adding Inspector Agent to sequence (Schema missing)")
            
            # Wrap MCP tool for ADK
            def inspect_dataset(script: str) -> str:
                """Runs a python script to inspect the dataframe 'df'. Returns the output."""
                print(f"🛠️ [Inspector] Executing MCP Tool 'run_script'...")
                result = client.call_tool("run_script", {"script": script})
                # Side effect: Save schema for future turns
                GLOBAL_MEMORY["schema_info"] = result 
                return result

            inspector_prompt = (
                "You are a Data Inspector. Your goal is to understand the dataset structure.\n"
                "The dataframe is ALREADY LOADED as variable 'df'.\n"
                "1. Use the `inspect_dataset` tool to run: `import io; df.info(); print(df.head())`\n"
                "2. Your final response MUST be the actual output of the script (the schema info)."
            )
            inspector = LlmAgent(name="inspector", model=SambaNovaLiteLlm(model=f"sambanova/{MODEL_NAME}"), instruction=inspector_prompt, tools=[inspect_dataset])
            agents.append(inspector)
            planner_context = "DATASET CONTEXT: The previous agent has provided the schema.\n\n"
        else:
            planner_context = f"DATASET CONTEXT:\n{schema_info}\n\n"
        
        # 2. Planner Agent
        planner_prompt = (
            "You are a Lead Data Analyst. Your job is to PLAN the analysis, not write the code.\n"
            f"{planner_context}"
            "TASK:\n"
            "1. Analyze the request and the dataset schema.\n"
            "2. Identify which columns are relevant.\n"
            "3. Propose a logical step-by-step plan:\n"
            "   - Data Cleaning (if needed)\n"
            "   - Feature Engineering (if needed)\n"
            "   - Specific Visualizations (Type, X, Y)\n"
            "   - Statistical Summaries\n"
            "4. Output a clear, numbered list."
        )
        planner = LlmAgent(name="planner", model=SambaNovaLiteLlm(model=f"sambanova/{MODEL_NAME}"), instruction=planner_prompt)
        agents.append(planner)
        
        # 3. Coder Agent
        # Note: In SequentialAgent, the output of the previous agent is passed to the next.
        # We add instructions to follow the plan.
        coder_prompt_seq = coder_system_prompt + "\n\nIMPORTANT: The previous message contains the ANALYSIS PLAN. Follow it step-by-step."
        coder = LlmAgent(name="coder", model=SambaNovaLiteLlm(model=f"sambanova/{MODEL_NAME}"), instruction=coder_prompt_seq)
        agents.append(coder)
        
        # 4. Create Sequential Agent
        root_agent = SequentialAgent(
            name="sequential_analyst",
            description="Sequential agent that runs the Inspector, Planner, and Coder agents in order.",
            sub_agents=agents
        )
        
        try:
            # Run the sequence
            session_service = InMemorySessionService()
            runner = Runner(agent=root_agent, app_name="data_analyst_pipeline", session_service=session_service)
            session = await session_service.create_session(app_name="data_analyst_pipeline", user_id="user")
            
            input_msg = types.Content(role="user", parts=[types.Part(text=f"User Request: {task_description}")])
            final_text = ""
            
            async for event in runner.run_async(session_id=session.id, user_id="user", new_message=input_msg):
                # Debug: Print intermediate thinking or content
                if hasattr(event, 'content') and event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(f"DEBUG: [Agent Stream] {part.text[:200]}...", file=sys.stderr)
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = event.content.parts[0].text
            
            return final_text.replace("```python", "").replace("```", "").strip()
        except Exception as e:
            print(f"❌ Sequential Agent Error: {e}")
            return ""

def generate_session_title(task_description, filename):
    """Generates a short title for the session using LLM."""
    prompt = (
        f"Generate a concise, professional title (3-6 words) for a data analysis session.\n"
        f"User Query: {task_description}\n"
        f"File: {filename}\n"
        f"Title:"
    )
    
    async def _run_title():
        try:
            response = await litellm.acompletion(
                model=f"sambanova/{MODEL_NAME}",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception:
            return "Analysis Session"

    try:
        title = asyncio.run(_run_title())
        return title.strip().strip('"')
    except Exception:
        return None

async def summarize_analysis(raw_output, task_description):
    """Uses an LLM to interpret the raw execution output and provide a user-friendly summary."""
    system_prompt = (
        "You are a Data Analyst. Your job is to interpret the raw output of a Python data analysis script.\n"
        "1. Explain the key findings based on the printed output.\n"
        "2. If the output contains statistical numbers, contextualize them.\n"
        "3. Do NOT mention 'the script printed' or 'raw output'. Present it as a final report.\n"
        "4. Keep it professional and concise.\n"
        "1. The output contains markers like [[IMG_0]], [[IMG_1]] representing generated graphs.\n"
        "2. You MUST preserve these markers in your output.\n"
        "3. Place the markers [[IMG_x]] immediately after the analysis text that describes that specific graph.\n"
        "4. Explain the key findings based on the printed output.\n"
        "5. If the output contains statistical numbers, contextualize them.\n"
        "6. Do NOT mention 'the script printed' or 'raw output'. Present it as a final report.\n"
        "7. The script usually prints data summaries before the image marker. Use this data to describe the graph accurately.\n"
    )
    
    user_content = f"User Question: {task_description}\n\nRaw Script Output:\n{raw_output}"
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        response = await litellm.acompletion(model=f"sambanova/{MODEL_NAME}", messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Summarizer Error: {e}")
        return raw_output

def test_direct_call(file_path, question, session_id):
    """The Loop Agent Orchestrator."""
    print(f"\n🔄 LOOP AGENT: Session {session_id[:8]}...")
    
    # Initialize MCP Client
    client = DirectMCPClient()
    
    try:
        client.start_server()

        # Logic to handle file loading and persistence
        # We must reload the CSV into the new MCP process every time
        current_csv = file_path if file_path else GLOBAL_MEMORY.get("current_csv")
        
        if current_csv:
            if file_path: # Only convert if explicitly uploaded
                file_ext = Path(current_csv).suffix.lower()
                if file_ext in ['.xlsx', '.xls']:
                    current_csv, _ = convert_excel_to_csv(current_csv)
                    session_manager.save_artifact(session_id, "converted_csv", Path(current_csv).name, current_csv)
                # New file uploaded: Clear old schema to force re-inspection
                GLOBAL_MEMORY["schema_info"] = None
            
            # 1. Always load CSV to ensure 'df' exists in this session
            print(f"1️⃣ Loading CSV via MCP: {current_csv}")
            client.call_tool("load_csv", {"csv_path": current_csv, "df_name": "df"})
            GLOBAL_MEMORY["current_csv"] = current_csv
            
        else:
             return {"text": "Please upload a CSV file to start analysis.", "images": [], "session_id": session_id}

        # Autoname Session if it has a default title
        current_sess = session_manager.get_session(session_id)
        if current_sess:
            current_title = current_sess.get("title", "")
            if current_title.startswith("Analysis of") or current_title == "New Analysis Session":
                fname = Path(current_csv).name if current_csv else "Data"
                
                new_title = generate_session_title(question, fname)
                if new_title:
                    session_manager.update_session_title(session_id, new_title)
                    print(f"🏷️ Session Renamed: {new_title}")

        # Loop Logic
        max_retries = 3
        attempt = 0
        current_code = ""
        last_error = None
        
        while attempt < max_retries:
            attempt += 1
            print(f"\n📢 Attempt {attempt}/{max_retries}")
            
            if attempt == 1:
                # Run Sequential Pipeline (Plan -> Code)
                current_code = asyncio.run(run_analysis_pipeline(client, question))
            else:
                # Run Fixer (Single Agent)
                current_code = asyncio.run(run_analysis_pipeline(client, question, previous_error=last_error, previous_code=current_code))
                
            print(f"DEBUG: [Generated Code to Execute]\n{current_code}\n-------------------", file=sys.stderr)
            
            # 3. Execute via MCP
            print(f"3️⃣ Executing script via MCP...")
            result_text = client.call_tool("run_script", {"script": current_code})
            
            # Check for explicit MCP error OR Python traceback/common errors in output
            is_error = "Error:" in result_text or "Traceback" in result_text or "NameError" in result_text or "SyntaxError" in result_text or "TypeError" in result_text or "ValueError" in result_text
            error_keywords = [
                "Error:", "Traceback", "NameError", "SyntaxError", "TypeError", "ValueError", 
                "AttributeError", "KeyError", "IndexError", "ImportError", "ModuleNotFoundError",
                "is not defined", "object has no attribute"
            ]
            is_error = any(keyword in result_text for keyword in error_keywords)
            
            if not is_error:
                # Parse Base64 images from stdout
                images = []
                clean_text_lines = []
                for line in result_text.split('\n'):
                    if "IMAGE_BASE64:" in line:
                        try:
                            # Extract base64 string
                            parts = line.split("IMAGE_BASE64:")
                            if len(parts) > 1:
                                b64_str = parts[1].strip()
                                img_data = base64.b64decode(b64_str)
                                
                                img_dir = Path(__file__).parent.parent / "img"
                                img_dir.mkdir(exist_ok=True)
                                fname = img_dir / f"plot_{session_id[:8]}_{len(images)}.png"
                                
                                with open(fname, "wb") as f:
                                    f.write(img_data)
                                images.append(b64_str)
                                clean_text_lines.append(f"[[IMG_{len(images)-1}]]")
                                print(f"🖼️ Saved image via MCP: {fname}")
                        except Exception as e:
                            print(f"⚠️ Failed to decode image: {e}")
                    else:
                        clean_text_lines.append(line)
                
                clean_output = "\n".join(clean_text_lines)
                print("4️⃣ Generating final summary...")
                summary = asyncio.run(summarize_analysis(clean_output, question))
                
                print(f"✅ Success on attempt {attempt}!")
                return {"status": "success", "text": summary, "images": images, "session_id": session_id}
            else:
                last_error = result_text
                print(f"⚠️ Attempt {attempt} failed.")
        
        return {"text": f"Failed after {max_retries} attempts.\nLast error: {last_error}", "images": [], "session_id": session_id}

    except Exception as e:
        print(f"❌ Error in test_direct_call: {e}")
        return {"text": f"Error: {e}", "images": [], "session_id": session_id}
    finally:
        client.close()