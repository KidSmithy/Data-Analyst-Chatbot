import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import time
import base64
import shutil
import tempfile
import traceback
import contextlib
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Important for backend
import matplotlib.pyplot as plt
from datetime import datetime
from sambanova import SambaNova

# Internal Imports
from config import GLOBAL_MEMORY, MODEL_NAME
from database import session_manager
from utils import convert_excel_to_csv
from pathlib import Path

def generate_code(task_description, session_id, previous_error=None, previous_code=None):
    """Generates code. If previous_error is provided, acts as 'Fixer'."""
    schema = GLOBAL_MEMORY["schema_info"]
    csv_path = GLOBAL_MEMORY["current_csv"]
    
    system_prompt = (
        f"You are a Python Data Analyst. Write code to analyze this CSV file: {csv_path}\n"
        f"DATA INFO: {schema[:500]}\n\n"
        "RULES:\n"
        "1. Load the CSV with pandas.\n"
        "2. Save all plots as PNG files in the current directory.\n"
        "3. Print 'IMAGE_SAVED: filename.png' after saving a plot.\n"
        "4. Print findings clearly.\n"
        "5. Output ONLY valid Python code. No markdown formatting."
    )

    if previous_error:
        print(f"🔧 [Coder Agent] Entering FIX mode...")
        user_content = (
            f"I tried to run your previous code, but it failed.\n\n"
            f"YOUR PREVIOUS CODE:\n{previous_code}\n\n"
            f"THE ERROR MESSAGE:\n{previous_error}\n\n"
            f"Please fix the code to resolve this error. Return the FULL corrected script."
        )
    else:
        print(f"🤖 [Coder Agent] Generating initial code...")
        user_content = f"Task: {task_description}"

    client = SambaNova(
        api_key=os.getenv("SAMBANOVA_API_KEY"), 
        base_url="https://api.sambanova.ai/v1",
    )
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_content}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.1, max_tokens=4000
        )
        code = response.choices[0].message.content
        return code.replace("```python", "").replace("```", "").strip()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return ""

def execute_script_locally(code, csv_path, session_id):
    """Executes code and captures output/images."""
    temp_dir = tempfile.mkdtemp(prefix="viz_")
    original_dir = os.getcwd()
    output_capture = io.StringIO()
    
    try:
        os.chdir(temp_dir)
        exec_globals = {'pd': pd, 'plt': plt, 'os': os, 'csv_path': csv_path}
        
        with contextlib.redirect_stdout(output_capture):
            exec(code, exec_globals)
            
        output_text = output_capture.getvalue()
        
        # Image Extraction
        image_files = []
        lines = output_text.split('\n')
        clean_lines = []
        
        for line in lines:
            if line.startswith('IMAGE_SAVED:'):
                filename = line.replace('IMAGE_SAVED:', '').strip()
                if os.path.exists(filename):
                    # Copy to img/ folder
                    try:
                        img_dir = Path(__file__).parent.parent / "img"
                        img_dir.mkdir(exist_ok=True)
                        dest_path = img_dir / f"plot_{session_id[:8]}_{len(image_files)}.png"
                        shutil.copy(filename, dest_path)
                    except Exception as e: print(f"⚠️ Failed to save image locally: {e}")

                    image_files.append(filename)
                    session_manager.save_artifact(
                        session_id, "visualization", filename, os.path.join(temp_dir, filename),
                        {"created_at": datetime.now().isoformat()}
                    )
            else:
                clean_lines.append(line)
        
        # Catch loose PNGs
        for f in os.listdir(temp_dir):
            if f.endswith('.png') and f not in image_files:
                # Copy to img/ folder
                try:
                    img_dir = Path(__file__).parent.parent / "img"
                    img_dir.mkdir(exist_ok=True)
                    dest_path = img_dir / f"plot_{session_id[:8]}_{len(image_files)}.png"
                    shutil.copy(os.path.join(temp_dir, f), dest_path)
                except Exception: pass

                image_files.append(f)
                session_manager.save_artifact(
                        session_id, "visualization", f, os.path.join(temp_dir, f),
                        {"created_at": datetime.now().isoformat()}
                )

        # Base64 Convert
        image_base64_strings = []
        for img_path in image_files:
            try:
                full_path = os.path.join(temp_dir, img_path) if not os.path.isabs(img_path) else img_path
                with open(full_path, "rb") as img_file:
                    image_base64_strings.append(base64.b64encode(img_file.read()).decode('utf-8'))
            except Exception: pass

        if len(output_text) > 20:
            session_manager.save_finding(session_id, "execution_output", output_text[:500], "code_output")

        return {"status": "success", "text": "\n".join(clean_lines), "images": image_base64_strings, "session_id": session_id}

    except Exception as e:
        return {"status": "error", "error_message": f"{str(e)}\n{traceback.format_exc()}", 
                "text": "", "images": [], "session_id": session_id}
    finally:
        os.chdir(original_dir)

def generate_session_title(task_description, filename):
    """Generates a short title for the session using LLM."""
    client = SambaNova(
        api_key=os.getenv("SAMBANOVA_API_KEY"), 
        base_url="https://api.sambanova.ai/v1",
    )
    
    prompt = (
        f"Generate a concise, professional title (3-6 words) for a data analysis session.\n"
        f"User Query: {task_description}\n"
        f"File: {filename}\n"
        f"Title:"
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception:
        return None

def test_direct_call(file_path, question, session_id):
    """The Loop Agent Orchestrator."""
    print(f"\n🔄 LOOP AGENT: Session {session_id[:8]}...")
    
    # Preparation
    try:
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.xlsx', '.xls']:
                file_path, _ = convert_excel_to_csv(file_path)
                session_manager.save_artifact(session_id, "converted_csv", Path(file_path).name, file_path)
            
            df = pd.read_csv(file_path)
            GLOBAL_MEMORY["current_csv"] = file_path
            
            preview = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\nColumns: {list(df.columns)[:10]}"
            GLOBAL_MEMORY["schema_info"] = f"Shape: {df.shape}, Columns: {list(df.columns)}"
            session_manager.update_session_data(session_id, preview, f"Dataset shape: {df.shape}")
            print(f"✅ CSV loaded. Shape: {df.shape}")
    except Exception as e:
        return {"text": f"Error loading file: {e}", "images": [], "session_id": session_id}

    # Autoname Session if it has a default title
    try:
        current_sess = session_manager.get_session(session_id)
        if current_sess:
            current_title = current_sess.get("title", "")
            if current_title.startswith("Analysis of") or current_title == "New Analysis Session":
                fname = Path(file_path).name if file_path else "Data"
                if not file_path and GLOBAL_MEMORY.get("current_csv"):
                     fname = Path(GLOBAL_MEMORY["current_csv"]).name
                
                new_title = generate_session_title(question, fname)
                if new_title:
                    session_manager.update_session_title(session_id, new_title)
                    print(f"🏷️ Session Renamed: {new_title}")
    except Exception as e:
        print(f"⚠️ Autoname failed: {e}")

    # Loop Logic
    max_retries = 3
    attempt = 0
    current_code = ""
    last_error = None
    
    if not file_path and GLOBAL_MEMORY["current_csv"]:
        file_path = GLOBAL_MEMORY["current_csv"]
    
    while attempt < max_retries:
        attempt += 1
        print(f"\n📢 Attempt {attempt}/{max_retries}")
        
        if attempt == 1:
            current_code = generate_code(question, session_id)
        else:
            current_code = generate_code(question, session_id, last_error, current_code)
            
        result = execute_script_locally(current_code, GLOBAL_MEMORY["current_csv"], session_id)
        
        if result["status"] == "success":
            print(f"✅ Success on attempt {attempt}!")
            return result
        else:
            last_error = result["error_message"]
            print(f"⚠️ Attempt {attempt} failed.")
            if attempt == max_retries:
                return {"text": f"Failed after {max_retries} attempts.\nLast error: {last_error}", 
                        "images": [], "session_id": session_id}

    return {"text": "Loop failed unexpectedly.", "images": [], "session_id": session_id}
