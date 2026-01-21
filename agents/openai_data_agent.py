import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import re
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
from config import GLOBAL_MEMORY
from database import session_manager
from utils import convert_excel_to_csv
from pathlib import Path

# --- OpenAI Configuration ---
# Fix typo in .env if present (OPENAI_APi_KEY -> OPENAI_API_KEY)
if "OPENAI_API_KEY" not in os.environ and "OPENAI_APi_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_APi_KEY"]

# Using gpt-4o as the latest standard model. 
# If you have access to specific beta models like gpt-5.2, change this string.
MODEL_GENERAL = "gpt-5.2"
MODEL_CODER = "gpt-5.1-codex-max"

# Configure LiteLLM
litellm.drop_params = True

# Suppress Pydantic warnings from LiteLLM
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

# --- Direct MCP Client ---
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
        print(f"🛠️ [MCP] Calling tool '{tool_name}' with args: {str(arguments)}...", file=sys.stderr)
        
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

def load_prompts():
    """Loads agent instructions from agents.md."""
    path = Path(__file__).parent / "agents.md"
    prompts = {}
    if path.exists():
        content = path.read_text(encoding="utf-8")
        sections = re.split(r'^#\s+(.+)$', content, flags=re.MULTILINE)
        for i in range(1, len(sections), 2):
            title = sections[i].strip().lower()
            body = sections[i+1].strip()
            prompts[title] = body
    return prompts

async def run_analysis_pipeline(client, task_description, previous_error=None, previous_code=None):
    """
    Orchestrates the analysis using Google ADK Agents with OpenAI.
    """
    csv_path = GLOBAL_MEMORY.get("current_csv", "data.csv")
    schema_info = GLOBAL_MEMORY.get("schema_info")
    
    # Knowledge Base features (Plans, Snippets, Docs) have been removed.
    examples_context = ""
    docs_context = ""
    plans_context = ""

    # Load Prompts
    prompts = load_prompts()
    
    # Prepare Coder Prompt
    coder_template = prompts.get("coder", "You are a Data Analyst.")
    coder_system_prompt = coder_template.replace("{{csv_path}}", str(csv_path)) \
                                        .replace("{{schema_info}}", str(schema_info)) \
                                        .replace("{{docs_context}}", str(docs_context)) \
                                        .replace("{{examples_context}}", str(examples_context))

    if previous_error:
        # --- FIX MODE: Single Agent ---
        print(f"🔧 [Coder Agent] Entering FIX mode...")
        user_content = (
            f"I tried to run your previous code, but it failed.\n\n"
            f"YOUR PREVIOUS CODE:\n{previous_code}\n\n"
            f"THE ERROR MESSAGE:\n{previous_error}\n\n"
            f"Please fix the code to resolve this error. Return the FULL, SELF-CONTAINED corrected script that defines all necessary variables."
        )
        
        try:
            messages = [
                {"role": "system", "content": coder_system_prompt},
                {"role": "user", "content": user_content}
            ]
            response = await litellm.acompletion(model=MODEL_CODER, messages=messages)
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

            inspector_prompt = prompts.get("inspector", "You are a Data Inspector.")
            # Use standard LiteLlm for OpenAI
            inspector = LlmAgent(name="inspector", model=LiteLlm(model=MODEL_CODER), instruction=inspector_prompt, tools=[inspect_dataset])
            agents.append(inspector)
            planner_context = "DATASET CONTEXT: The previous agent has provided the schema.\n\n"
        else:
            planner_context = f"DATASET CONTEXT:\n{schema_info}\n\n"
        
        # 2. Planner Agent
        planner_template = prompts.get("planner", "You are a Planner.")
        planner_prompt = planner_template.replace("{{planner_context}}", str(planner_context)) \
                                         .replace("{{plans_context}}", str(plans_context))
        
        planner = LlmAgent(name="planner", model=LiteLlm(model=MODEL_GENERAL), instruction=planner_prompt)
        agents.append(planner)
        
        # 3. Coder Agent
        coder_prompt_seq = coder_system_prompt + "\n\nIMPORTANT: The previous message contains the ANALYSIS PLAN. Follow it step-by-step."
        coder = LlmAgent(name="coder", model=LiteLlm(model=MODEL_CODER), instruction=coder_prompt_seq)
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
    prompt = f"Generate a concise title (3-6 words) for: {task_description} on file {filename}"
    
    async def _run_title():
        try:
            response = await litellm.acompletion(
                model=MODEL_GENERAL,
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
    prompts = load_prompts()
    system_prompt = prompts.get("summarizer", "You are a Summarizer.")
    
    user_content = f"User Question: {task_description}\n\nRaw Script Output:\n{raw_output}"
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        response = await litellm.acompletion(model=MODEL_GENERAL, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Summarizer Error: {e}")
        return raw_output

def test_direct_call(file_path, question, session_id):
    """The Loop Agent Orchestrator."""
    print(f"\n🔄 OPENAI LOOP AGENT: Session {session_id[:8]}...")
    
    client = DirectMCPClient()
    
    try:
        client.start_server()

        current_csv = file_path if file_path else GLOBAL_MEMORY.get("current_csv")
        
        if current_csv:
            if file_path:
                file_ext = Path(current_csv).suffix.lower()
                if file_ext in ['.xlsx', '.xls']:
                    current_csv, _ = convert_excel_to_csv(current_csv)
                    session_manager.save_artifact(session_id, "converted_csv", Path(current_csv).name, current_csv)
                GLOBAL_MEMORY["schema_info"] = None
            
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
                current_code = asyncio.run(run_analysis_pipeline(client, question))
            else:
                current_code = asyncio.run(run_analysis_pipeline(client, question, previous_error=last_error, previous_code=current_code))
                
            print(f"DEBUG: [Generated Code to Execute]\n{current_code}\n-------------------", file=sys.stderr)
            
            print(f"3️⃣ Executing script via MCP...")
            result_text = client.call_tool("run_script", {"script": current_code})
            
            # Check for explicit MCP error OR Python traceback/common errors in output
            error_keywords = [
                "Error:", "Traceback", "NameError", "SyntaxError", "TypeError", "ValueError", 
                "AttributeError", "KeyError", "IndexError", "ImportError", "ModuleNotFoundError",
                "is not defined", "object has no attribute"
            ]
            is_error = any(keyword in result_text for keyword in error_keywords)
            
            if not is_error:
                # Image extraction logic (same as before)
                images = []
                raw_output_with_placeholders = ""
                
                for line in result_text.split('\n'):
                    if "IMAGE_BASE64:" in line:
                        try:
                            parts = line.split("IMAGE_BASE64:")
                            if len(parts) > 1:
                                b64_str = parts[1].strip()
                                
                                # Save image to img/ folder
                                try:
                                    img_data = base64.b64decode(b64_str)
                                    img_dir = Path(__file__).parent.parent / "img"
                                    img_dir.mkdir(exist_ok=True)
                                    fname = img_dir / f"plot_{session_id[:8]}_{len(images)}.png"
                                    with open(fname, "wb") as f:
                                        f.write(img_data)
                                    print(f"🖼️ Saved image: {fname}")
                                except Exception as e:
                                    print(f"⚠️ Failed to save image: {e}")
                                
                                raw_output_with_placeholders += f"\n[[IMG_{len(images)}]]\n"
                                images.append(b64_str)
                        except Exception: pass
                    else:
                        raw_output_with_placeholders += line + "\n"
                
                print("4️⃣ Generating final summary...")
                summary = asyncio.run(summarize_analysis(raw_output_with_placeholders, question))
                
                # Replace placeholders with HTML
                final_html = summary
                used_indices = []
                for idx, img_b64 in enumerate(images):
                    marker = f"[[IMG_{idx}]]"
                    html_tag = f'<img src="data:image/png;base64,{img_b64}" style="max-width: 100%; margin: 10px 0; border-radius: 8px;">'
                    if marker in final_html:
                        final_html = final_html.replace(marker, html_tag)
                        used_indices.append(idx)
                
                unused_images = [img for i, img in enumerate(images) if i not in used_indices]
                if unused_images:
                    final_html += "\n\n" + "\n".join([f'<img src="data:image/png;base64,{img}" style="max-width: 100%; margin: 10px 0; border-radius: 8px;">' for img in unused_images])

                return {"status": "success", "text": final_html, "images": [], "session_id": session_id, "code": current_code}
            else:
                last_error = result_text
                print(f"⚠️ Attempt {attempt} failed.")

        return {"text": f"Failed after {max_retries} attempts.\nLast error: {last_error}", "images": [], "session_id": session_id}

    except Exception as e:
        print(f"❌ Error in test_direct_call: {e}")
        return {"text": f"Error: {e}", "images": [], "session_id": session_id}
    finally:
        client.close()