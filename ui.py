import gradio as gr
import sys
import asyncio
from pathlib import Path
from config import GLOBAL_MEMORY
from database import session_manager

# Import Agents
# Ensure you have an empty __init__.py in the agents/ folder
from agents import data_agent 
from agents.research_agent import ResearchAgent

# Global instance for the Research Agent (injected from main.py)
research_agent_instance = None 

def set_research_agent(agent):
    """Called by main.py to inject the initialized Research Agent."""
    global research_agent_instance
    research_agent_instance = agent
    
    # Attempt to register the database tool dynamically
    # This allows the AI to use the function we defined in database.py
    if hasattr(agent, "register_tool"):
        try:
            agent.register_tool(session_manager.ai_search_papers)
            print("✅ Registered ai_search_papers tool with ResearchAgent", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Could not register tool: {e}", file=sys.stderr)
    elif hasattr(agent, "tools") and isinstance(agent.tools, list):
        agent.tools.append(session_manager.ai_search_papers)
        print("✅ Added ai_search_papers to ResearchAgent tools list", file=sys.stderr)

def update_input_visibility(mode, interactive=True):
    """Switches the input box style based on the selected agent."""
    if mode == "🔬 Research Assistant":
        # Hide file upload, change placeholder
        return gr.MultimodalTextbox(value=None, file_count="none", placeholder="Enter your research query (e.g., 'Latest papers on LLM agents')...", interactive=interactive)
    else:
        # Show file upload, default placeholder
        return gr.MultimodalTextbox(value=None, file_count="single", placeholder="Upload a CSV/Excel file or ask a data question...", interactive=interactive)

def get_history_html():
    """Generates HTML list of sessions with delete buttons."""
    try:
        sessions = session_manager.list_sessions(limit=1000)
        
        html = """
        <style>
            .session-item { display: flex; justify-content: space-between; align-items: center; padding: 8px; cursor: pointer; border-bottom: 1px solid #eee; border-radius: 4px; margin-bottom: 4px; }
            .session-item:hover { background-color: #f5f5f5; }
            .session-info { flex-grow: 1; overflow: hidden; }
            .session-title { font-weight: 600; font-size: 0.95em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
            .session-date { font-size: 0.75em; color: #888; }
            .delete-icon { cursor: pointer; padding: 4px 8px; border-radius: 4px; opacity: 0.6; }
            .delete-icon:hover { background-color: #ffdddd; opacity: 1; }
        </style>
        <div class='session-list' style='max-height: 600px; overflow-y: auto;'>
        """
        
        for s in sessions:
            sid = s['session_id']
            title = s['title'] or "Untitled"
            date = s['created_at'][:10] if s['created_at'] else ""
            safe_title = title.replace("'", "&#39;").replace('"', "&quot;")
            
            html += f"""
            <div class='session-item' onclick="handleSessionAction('load', '{sid}')">
                <div class='session-info'>
                    <div class='session-title'>{safe_title}</div>
                    <div class='session-date'>{date}</div>
                </div>
                <div class='delete-icon' onclick="event.stopPropagation(); handleSessionAction('delete', '{sid}')" title="Delete Session">
                    🗑️
                </div>
            </div>
            """
        html += "</div>"
        return html
    except Exception as e:
        return f"<div>Error loading sessions: {e}</div>"
    

def load_session_history(selected_session_id):
    """Loads history for a selected session from DB."""
    print(f"DEBUG: [ui.py] load_session_history called for ID: {selected_session_id}", file=sys.stderr)
    if not selected_session_id:
        return [], [], None, None
    
    try:
        # Fetch messages (increase limit to see full context)
        db_messages = session_manager.get_messages(selected_session_id, limit=100)
        
        chat_history = []
        internal_history = []
        
        # Convert DB format to Gradio Messages format (List of Dicts)
        for msg in db_messages:
            role = msg["role"]
            content = msg["content"]
            # Ensure strict format: {"role": "user"|"assistant", "content": "..."}
            entry = {"role": str(role).lower(), "content": str(content)}
            if "generated_code" in msg:
                entry["code"] = msg["generated_code"]
            chat_history.append(entry)
            internal_history.append(entry)
            
        return chat_history, internal_history, selected_session_id, selected_session_id
    except Exception as e:
        print(f"Error loading session: {e}")
        return [], [], None, None

def user_submit(message_dict, history, internal_history, session_state, agent_mode):
    """
    Step 1: Handle User Input
    - Saves user message to DB
    - Updates Chat UI with user message
    - Prepares state for the Bot Response step
    """
    text = message_dict.get("text", "").strip()
    files = message_dict.get("files", [])
    file_path = files[0] if files else None
    
    # 1. Session Management
    if not session_state:
        session_id = session_manager.create_session(file_path)
        session_state = session_id
        GLOBAL_MEMORY["current_session_id"] = session_id
        
        # Rename session using the first prompt if available
        if text:
            # Create a title from the first ~50 characters of the prompt
            new_title = text[:50].strip()
            if len(text) > 50:
                new_title += "..."
            session_manager.update_session_title(session_id, new_title)
    else:
        session_id = session_state
    
    # 2. Save User Message
    if text:
        session_manager.save_message(session_id, "user", text)
    
    # 3. Update UI History
    display_text = text
    if file_path and agent_mode == "📊 Data Analyst":
        display_text += f" (File: {Path(file_path).name})"
    
    history.append({"role": "user", "content": display_text})
    internal_history.append({"role": "user", "content": display_text})
    
    # 4. Add "Thinking..." placeholder
    wait_msg = "⏳ Analyzing Data..." if agent_mode == "📊 Data Analyst" else "⏳ Researching (Querying Scholar APIs)..."
    history.append({"role": "assistant", "content": wait_msg})
    
    # 5. Prepare Data for Next Step (Bot Respond)
    input_data = {
        "text": text, 
        "file": file_path, 
        "session_id": session_id, 
        "mode": agent_mode
    }
    
    # Return: Clear Input Box, Update History, Update Internal Hist, Pass Data, Pass Session
    # Use update_input_visibility to maintain correct file_count while disabled
    return update_input_visibility(agent_mode, interactive=False), history, internal_history, input_data, session_state, session_id

async def bot_respond(history, internal_history, session_state, input_data):
    """
    Step 2: Generate Bot Response (The Router)
    - Routes request to Data Agent or Research Agent
    - Updates UI with final response
    """
    global research_agent_instance
    text = input_data.get("text")
    file_path = input_data.get("file")
    session_id = input_data.get("session_id")
    mode = input_data.get("mode", "📊 Data Analyst")
    
    generated_code = None
    full_response = ""
    
    try:
        # --- ROUTER LOGIC ---
        if mode == "📊 Data Analyst":
            # Run Data Agent
            # CRITICAL: Run sync code in a thread to avoid blocking the async event loop
            if file_path or GLOBAL_MEMORY["current_csv"]:
                result = await asyncio.to_thread(data_agent.test_direct_call, file_path, text, session_id)
            else:
                result = {"text": "Please upload a file first to start the analysis.", "images": [], "session_id": session_id}
            
            # Format Data Response (Text + Images)
            response_content = result["text"]
            generated_code = result.get("code")
            images = result["images"]
            
            # Replace markers [[IMG_x]] with actual HTML images
            for idx, img in enumerate(images):
                marker = f"[[IMG_{idx}]]"
                html_img = f'<img src="data:image/png;base64,{img}" style="max-width: 100%; margin: 10px 0; border-radius: 8px;">'
                if marker in response_content:
                    response_content = response_content.replace(marker, html_img)
                else:
                    # Fallback: Append if marker is missing
                    response_content += f"\n\n{html_img}"
            
            full_response = response_content

        elif mode == "🔬 Research Assistant":
            # Run Research Agent (MCP)
            # Auto-initialize if not injected (e.g. running ui.py directly)
            if not research_agent_instance:
                print("🔄 Initializing Research Agent...")
                research_agent_instance = ResearchAgent()

            if research_agent_instance:
                # This is already async, so we await it directly
                response_data = await research_agent_instance.process_message(text, session_id)
                
                print(f"DEBUG: [ui.py] Received response from ResearchAgent. Type: {type(response_data)}", file=sys.stderr)
                if isinstance(response_data, dict):
                    print(f"DEBUG: [ui.py] Response keys: {list(response_data.keys())}", file=sys.stderr)
                
                # Handle structured response (Text + Papers)
                if isinstance(response_data, dict) and "text" in response_data:
                    full_response = response_data["text"]
                    papers = response_data.get("papers", [])
                    if papers:
                        print(f"DEBUG: Saving {len(papers)} papers from research agent...", file=sys.stderr)
                        for paper in papers:
                            session_manager.save_paper(paper)
                        
                        # --- PARALLEL AGENT EXECUTION ---
                        # Launch the Deep Dive Agent in the background (Fire & Forget)
                        asyncio.create_task(research_agent_instance.deep_dive_into_papers(papers, session_id))
                else:
                    full_response = str(response_data)
            else:
                full_response = "⚠️ Error: Research Agent is not connected. Please check your API keys or server connection."
                
        # --- SAVE & UPDATE ---
        # Save the full text response to DB (Research outputs can be long)
        session_manager.save_message(session_id, "assistant", full_response)
        
        # Update Chat UI
        history[-1] = {"role": "assistant", "content": full_response}
        
        internal_entry = {"role": "assistant", "content": full_response}
        if generated_code:
            internal_entry["code"] = generated_code
        internal_history.append(internal_entry)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history[-1] = {"role": "assistant", "content": error_msg}
    
    # Reset input box based on current mode to maintain correct placeholder/file_count
    new_input = update_input_visibility(mode)
    return history, internal_history, new_input, session_state

def handle_session_action(action_value, current_session_id):
    """Handles actions from the HTML session list."""
    print(f"DEBUG: [ui.py] handle_session_action triggered with '{action_value}'", file=sys.stderr)
    if not action_value or ":" not in action_value:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""
        
    action, target_id = action_value.split(":", 1)
    
    if action == "load":
        chat, internal, state, disp = load_session_history(target_id)
        return gr.update(), chat, internal, state, disp, ""
        
    elif action == "delete":
        session_manager.delete_session(target_id)
        new_html = get_history_html()
        if target_id == current_session_id:
            return new_html, [], [], None, None, ""
        else:
            return new_html, gr.update(), gr.update(), gr.update(), gr.update(), ""
        
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""

JS_SESSION_HANDLER = """
function() {
    window.handleSessionAction = function(action, id) {
        console.log("DEBUG: JS handleSessionAction invoked.", {action: action, id: id});
        
        if (action === 'delete' || action === 'delete_paper') {
            if (!confirm('Are you sure you want to delete this item?')) {
                return;
            }
        }
        
        // Find the action bridge element
        var container = document.getElementById('action_bridge');
        if (!container) {
            console.error("DEBUG: #action_bridge container NOT found.");
            return;
        }
        
        // Try to find the textarea/input element
        var bridge = container.querySelector('textarea') || container.querySelector('input');
        if (!bridge) {
            console.error("DEBUG: Input/Textarea NOT found inside #action_bridge.");
            return;
        }
        
        var payload = action + ":" + id;
        bridge.value = payload;
        bridge.dispatchEvent(new Event('input', { bubbles: true }));
    };
    console.log("DEBUG: handleSessionAction function registered globally.");
}
"""

def create_ui():
    with gr.Blocks(title="AI Workstation") as demo:
        gr.HTML("""<style>
            #action_bridge { display: none; }
            .gradio-container { gap: 10px !important; }
        </style>""")

        # Inject the JavaScript function FIRST, before any HTML that uses it
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js=JS_SESSION_HANDLER
        )
        
        gr.Markdown("# 🤖 Integrated AI Workstation")
        
        # --- Sidebar for Session Management ---
        with gr.Sidebar(label="Session History") as sidebar:
            
            new_chat_btn = gr.Button("➕ New Chat", variant="primary")
            
            # HTML List for Sessions
            session_list = gr.HTML(value=get_history_html())
            
            # Hidden bridge for JS communication
            action_bridge = gr.Textbox(elem_id="action_bridge", visible=True, show_label=False)
            
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            session_id_display = gr.Textbox(label="Current Session ID", interactive=False)
        
        with gr.Tabs():
            with gr.Tab("💬 Workspace"):
                # --- 1. Top Control Bar (The Switch) ---
                with gr.Row(variant="panel"):
                    with gr.Column(scale=1):
                        agent_selector = gr.Dropdown(
                            choices=["📊 Data Analyst", "🔬 Research Assistant"], 
                            value="📊 Data Analyst", 
                            label="Select Active Agent",
                            interactive=True
                        )
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "**📊 Data Analyst Agent:**\n"
                            "Upload your data files (CSV/Excel). I automatically write and run Python code to clean, analyze, and visualize your data. You can ask for detailed breakdowns or quick questions like, 'What are the top 5 sales regions?'\n\n"
                            "**🔬 Research Assistant Agent:**\n"
                            "Ask me any question about recent topics. I will find and summarize relevant papers from academic databases like Google Scholar and Semantic Scholar to provide you with an expert explanation."
                        )

                # --- 2. Chat Interface ---
                chatbot = gr.Chatbot(height=600, label="Workspace")
                
                # Input box (Default: Data Analyst mode with file upload)
                chat_input = gr.MultimodalTextbox(
                    interactive=True, 
                    file_types=[".csv", ".xlsx", ".xls"], 
                    placeholder="Upload a file or ask a question...",
                    show_label=False
                )
                
                # Event: Change Input Box style when Agent switches
                agent_selector.change(fn=update_input_visibility, inputs=agent_selector, outputs=chat_input)

        # --- 3. State Management ---
        session_state = gr.State()
        internal_history = gr.State([])
        input_data_state = gr.State({}) # Holds data between 'user_submit' and 'bot_respond'

        # --- 4. Submission Logic (Chained) ---
        chat_input.submit(
            fn=user_submit, 
            inputs=[chat_input, chatbot, internal_history, session_state, agent_selector], 
            outputs=[chat_input, chatbot, internal_history, input_data_state, session_state, session_id_display]
        ).then(
            fn=bot_respond, 
            inputs=[chatbot, internal_history, session_state, input_data_state], 
            outputs=[chatbot, internal_history, chat_input, session_state]
        ) 
        
        # --- 5. Session Management Events ---
        def start_new_session():
            # Clears: chatbot, internal_history, session_state, session_id_display, action_bridge
            return [], [], None, None, ""
            
        new_chat_btn.click(
            fn=start_new_session,
            outputs=[chatbot, internal_history, session_state, session_id_display, action_bridge]
        )
        
        # Handle clicks from the HTML list
        action_bridge.change(
            fn=handle_session_action,
            inputs=[action_bridge, session_state],
            outputs=[session_list, chatbot, internal_history, session_state, session_id_display, action_bridge]
        )
        
        # Also update the HTML when refreshing
        def refresh_sessions():
            return get_history_html(), ""
        
        refresh_btn.click(
            fn=refresh_sessions,
            outputs=[session_list, action_bridge]
        )
        
    return demo, sidebar