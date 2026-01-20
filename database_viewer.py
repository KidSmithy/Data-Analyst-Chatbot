import gradio as gr
import sys
import pandas as pd
from database import session_manager

def create_database_interface():
    """Creates the Gradio interface for database management."""
    
    with gr.Blocks() as db_interface:
        gr.Markdown("## 🗄️ Database Management")
        
        with gr.Tabs() as main_tabs:
            # ================= SESSIONS TAB =================
            with gr.Tab("Sessions", id="tab_sessions"):
                with gr.Row():
                    refresh_sess_btn = gr.Button("🔄 Refresh List")
                
                sessions_output = gr.Dataframe(
                    headers=["session_id", "title", "created_at", "file_name"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    label="Active Sessions",
                    wrap=True
                )
                
                sess_msg = gr.Markdown("")
                
                with gr.Group():
                    gr.Markdown("### ✏️ Rename Session")
                    with gr.Row():
                        rename_id_input = gr.Textbox(label="Session ID", placeholder="Paste ID")
                        rename_title_input = gr.Textbox(label="New Title", placeholder="Enter new title")
                        rename_btn = gr.Button("Update Title")
                    
                    def rename_session(sid, title):
                        if not sid or not title: return "⚠️ Missing ID or Title.", get_sessions()
                        try:
                            session_manager.update_session_title(sid, title)
                            return f"✅ Renamed to '{title}'", get_sessions()
                        except Exception as e:
                            return f"❌ Error: {e}", get_sessions()
                    
                    rename_btn.click(rename_session, inputs=[rename_id_input, rename_title_input], outputs=[sess_msg, sessions_output])

                with gr.Group():
                    with gr.Row():
                        sess_id_input = gr.Textbox(label="Paste Session ID to Delete")
                        delete_sess_btn = gr.Button("🗑️ Delete Session", variant="secondary",)
                    
                    with gr.Accordion("⚠️ Danger Zone", open=False):
                        delete_all_btn = gr.Button("🔥 DELETE ALL DATA", variant="stop")

                def get_sessions():
                    data = session_manager.get_all_sessions()
                    if not data: return pd.DataFrame(columns=["session_id", "title", "created_at", "file_name"])
                    df = pd.DataFrame(data)
                    # Ensure columns exist
                    cols = ["session_id", "title", "created_at", "file_name"]
                    for c in cols:
                        if c not in df.columns: df[c] = ""
                    return df[cols]

                def delete_session(sess_id):
                    if not sess_id: return "⚠️ Please enter a Session ID.", get_sessions()
                    try:
                        session_manager.delete_session(sess_id)
                        return f"✅ Session {sess_id} deleted.", get_sessions()
                    except Exception as e:
                        return f"❌ Error: {e}", get_sessions()

                def delete_all():
                    try:
                        session_manager.reset_database()
                        return "✅ All data deleted.", get_sessions()
                    except Exception as e:
                        return f"❌ Error: {e}", get_sessions()

                refresh_sess_btn.click(get_sessions, outputs=sessions_output)
                delete_sess_btn.click(delete_session, inputs=[sess_id_input], outputs=[sess_msg, sessions_output])
                delete_all_btn.click(delete_all, outputs=[sess_msg, sessions_output])
                
                # Load on init
                db_interface.load(get_sessions, outputs=sessions_output)

            # ================= FINDINGS TAB =================
            with gr.Tab("Findings (RAG Data)", id="tab_findings"):
                with gr.Row():
                    refresh_find_btn = gr.Button("🔄 Refresh Findings")
                    delete_find_btn = gr.Button("🗑️ Delete Finding by ID", variant="stop")
                
                findings_output = gr.Dataframe(
                    headers=["id", "session_title", "finding_type", "content", "created_at"],
                    interactive=False,
                    wrap=True
                )
                
                find_id_input = gr.Number(label="Finding ID", precision=0)
                find_msg = gr.Markdown("")

                def get_findings():
                    data = session_manager.get_all_findings()
                    if not data: return pd.DataFrame(columns=["id", "session_title", "finding_type", "content", "created_at"])
                    df = pd.DataFrame(data)
                    cols = ["id", "session_title", "finding_type", "content", "created_at"]
                    for c in cols:
                        if c not in df.columns: df[c] = ""
                    return df[cols]

                def delete_finding(fid):
                    if not fid: return "⚠️ Enter ID.", get_findings()
                    try:
                        session_manager.delete_finding(int(fid))
                        return f"✅ Finding {fid} deleted.", get_findings()
                    except Exception as e:
                        return f"❌ Error: {e}", get_findings()

                refresh_find_btn.click(get_findings, outputs=findings_output)
                delete_find_btn.click(delete_finding, inputs=[find_id_input], outputs=[find_msg, findings_output])

            # ================= PAPERS TAB =================
            with gr.Tab("Research Papers", id="tab_papers"):
                papers_state = gr.State()

                with gr.Row():
                    # Left Column: List
                    with gr.Column():
                        with gr.Row():
                            refresh_paper_btn = gr.Button("🔄 Refresh Papers", size="sm")
                        
                        papers_output = gr.Dataframe(
                            headers=["id", "title", "year", "authors", "source"],
                            interactive=False,
                            wrap=False,
                            label="Paper List (Click row to view details)"
                        )
                        
                        with gr.Row():
                            paper_id_input = gr.Number(label="Paper ID", precision=0)
                            delete_paper_btn = gr.Button("🗑️ Delete Paper", variant="stop", size="sm")
                            paper_msg = gr.Markdown("")

                    # Right Co  lumn: Details
                    with gr.Column():
                        gr.Markdown("### 📝 Edit Selected Paper")
                        with gr.Group():
                            # Hidden ID field to track which paper we are editing
                            edit_id = gr.Number(label="ID", visible=False, precision=0)
                            edit_title = gr.Textbox(label="Title", interactive=True)
                            with gr.Row():
                                edit_year = gr.Textbox(label="Year", interactive=True)
                                edit_source = gr.Textbox(label="Source", interactive=True)
                            edit_authors = gr.Textbox(label="Authors", interactive=True)
                            edit_abstract = gr.TextArea(label="Abstract (Edit to fix errors)", lines=10, interactive=True)
                            with gr.Row():
                                save_edit_btn = gr.Button("💾 Save Changes", variant="primary")
                                delete_edit_btn = gr.Button("🗑️ Delete This Paper", variant="stop")
                        
                        edit_msg = gr.Markdown("")
                
                gr.Markdown("### ➕ Add New Paper Manually")
                with gr.Group():
                    with gr.Row():
                        p_title = gr.Textbox(label="Title")
                        p_year = gr.Textbox(label="Year")
                    with gr.Row():
                        p_authors = gr.Textbox(label="Authors")
                        p_source = gr.Textbox(label="Source (e.g. PDF, URL)")
                    p_abstract = gr.Textbox(label="Abstract", lines=3)
                    add_paper_btn = gr.Button("Save Paper")

                def get_papers():
                    data = session_manager.get_all_papers()
                    if not data: 
                        return pd.DataFrame(columns=["id", "title", "year", "authors", "source"]), []
                    
                    df = pd.DataFrame(data)
                    full_data = df.to_dict('records')
                    
                    cols = ["id", "title", "year", "authors", "source"]
                    for c in cols:
                        if c not in df.columns: df[c] = ""
                    return df[cols], full_data

                def delete_paper(pid):
                    if not pid: 
                        df, full = get_papers()
                        return "⚠️ Enter ID.", df, full
                    try:
                        session_manager.delete_paper(int(pid))
                        df, full = get_papers()
                        return f"✅ Paper {pid} deleted.", df, full
                    except Exception as e:
                        df, full = get_papers()
                        return f"❌ Error: {e}", df, full
                
                def add_paper(title, year, authors, source, abstract):
                    if not title: 
                        df, full = get_papers()
                        return "⚠️ Title required.", df, full
                    try:
                        session_manager.save_paper({
                            "title": title, "year": year, "authors": authors,
                            "source": source, "abstract": abstract, "url": "", "raw_data": ""
                        })
                        df, full = get_papers()
                        return "✅ Paper added.", df, full
                    except Exception as e:
                        df, full = get_papers()
                        return f"❌ Error: {e}", df, full

                def display_paper_details(evt: gr.SelectData, all_data):
                    """Populates the edit fields when a row is clicked."""
                    if evt.index is None or not all_data:
                        return 0, 0, "", "", "", "", ""
                    try:
                        row_index = evt.index[0]
                        if row_index < len(all_data):
                            data = all_data[row_index]
                            
                            pid = data.get("id", 0)
                            title = data.get("title", "Untitled")
                            authors = data.get("authors", "Unknown")
                            year = data.get("year", "N/A")
                            source = data.get("source", "")
                            abstract = data.get("abstract", "No abstract available.")
                            
                            return pid, pid, title, year, source, authors, abstract
                    except Exception as e:
                        print(f"Error selecting paper: {e}")
                    return 0, 0, "", "", "", "", ""

                def save_paper_changes(pid, title, year, source, authors, abstract):
                    if not pid:
                        return "⚠️ No paper selected.", get_papers()[0], get_papers()[1]
                    try:
                        session_manager.update_paper_by_id(int(pid), title, year, authors, source, abstract)
                        df, full = get_papers()
                        return f"✅ Paper {pid} updated successfully.", df, full
                    except Exception as e:
                        return f"❌ Error updating: {e}", get_papers()[0], get_papers()[1]

                refresh_paper_btn.click(get_papers, outputs=[papers_output, papers_state])
                delete_paper_btn.click(delete_paper, inputs=[paper_id_input], outputs=[paper_msg, papers_output, papers_state])
                add_paper_btn.click(add_paper, inputs=[p_title, p_year, p_authors, p_source, p_abstract], outputs=[paper_msg, papers_output, papers_state])
                
                # When row selected -> Populate Edit Fields
                papers_output.select(display_paper_details, inputs=[papers_state], outputs=[paper_id_input, edit_id, edit_title, edit_year, edit_source, edit_authors, edit_abstract])
                
                # Save Button Action
                save_edit_btn.click(save_paper_changes, inputs=[edit_id, edit_title, edit_year, edit_source, edit_authors, edit_abstract], outputs=[edit_msg, papers_output, papers_state])
                delete_edit_btn.click(delete_paper, inputs=[edit_id], outputs=[edit_msg, papers_output, papers_state])

                # Load papers on launch
                db_interface.load(get_papers, outputs=[papers_output, papers_state])

            # ================= CODE LIBRARY TAB =================
            with gr.Tab("Code Library", id="tab_code"):
                with gr.Row():
                    refresh_code_btn = gr.Button("🔄 Refresh Snippets")
                
                code_output = gr.Dataframe(
                    headers=["id", "title", "description", "created_at"],
                    interactive=False,
                    wrap=True,
                    label="Saved Code Patterns"
                )
                
                with gr.Row():
                    code_id_input = gr.Number(label="Snippet ID", precision=0)
                    delete_code_btn = gr.Button("🗑️ Delete Snippet", variant="stop")
                
                code_msg = gr.Markdown("")

                gr.Markdown("### ➕ Add New Code Pattern")
                with gr.Group():
                    c_title = gr.Textbox(label="Title (e.g., 'Correlation Heatmap')")
                    c_desc = gr.Textbox(label="Description (Keywords for search)")
                    c_code = gr.Code(label="Python Code", language="python", lines=10)
                    add_code_btn = gr.Button("💾 Save Snippet")

                def get_snippets():
                    if not hasattr(session_manager, "get_all_code_snippets"):
                        return pd.DataFrame(columns=["id", "title", "description", "created_at"])
                    data = session_manager.get_all_code_snippets()
                    if not data: return pd.DataFrame(columns=["id", "title", "description", "created_at"])
                    df = pd.DataFrame(data)
                    cols = ["id", "title", "description", "created_at"]
                    for c in cols:
                        if c not in df.columns: df[c] = ""
                    return df[cols]

                def delete_snippet(sid):
                    if not sid: return "⚠️ Enter ID.", get_snippets()
                    try:
                        session_manager.delete_code_snippet(int(sid))
                        return f"✅ Snippet {sid} deleted.", get_snippets()
                    except Exception as e:
                        return f"❌ Error: {e}", get_snippets()

                def add_snippet(title, desc, code):
                    if not title or not code: return "⚠️ Title and Code required.", get_snippets()
                    try:
                        session_manager.save_code_snippet(title, code, desc)
                        return "✅ Snippet saved.", get_snippets()
                    except Exception as e:
                        return f"❌ Error: {e}", get_snippets()

                refresh_code_btn.click(get_snippets, outputs=code_output)
                delete_code_btn.click(delete_snippet, inputs=[code_id_input], outputs=[code_msg, code_output])
                add_code_btn.click(add_snippet, inputs=[c_title, c_desc, c_code], outputs=[code_msg, code_output])
                
                # Load on init
                db_interface.load(get_snippets, outputs=code_output)

            # ================= DOCUMENTATION TAB =================
            with gr.Tab("Library Docs", id="tab_docs"):
                with gr.Row():
                    refresh_docs_btn = gr.Button("🔄 Refresh Docs")
                    seed_defaults_btn = gr.Button("🌱 Load Default Docs")
                
                docs_output = gr.Dataframe(
                    headers=["id", "library_name", "topic", "created_at"],
                    interactive=False,
                    wrap=True,
                    label="Saved Documentation"
                )
                
                with gr.Row():
                    doc_id_input = gr.Number(label="Doc ID", precision=0)
                    delete_doc_btn = gr.Button("🗑️ Delete Doc", variant="stop")
                
                doc_msg = gr.Markdown("")

                gr.Markdown("### ➕ Add New Documentation")
                with gr.Group():
                    with gr.Row():
                        d_lib = gr.Textbox(label="Library (e.g., Pandas)")
                        d_topic = gr.Textbox(label="Topic (e.g., Merge)")
                    d_file = gr.File(label="Upload Text/Markdown File (Optional)", file_types=[".txt", ".md"])
                    d_content = gr.TextArea(label="Content (Or paste text here)", lines=10)
                    add_doc_btn = gr.Button("💾 Save Documentation")

                def load_file_content(file):
                    if file is None: return ""
                    try:
                        with open(file.name, 'r', encoding='utf-8') as f: return f.read()
                    except Exception as e: return f"Error reading file: {e}"

                def get_docs():
                    if not hasattr(session_manager, "get_all_documentation"): return pd.DataFrame(columns=["id", "library_name", "topic", "created_at"])
                    data = session_manager.get_all_documentation()
                    if not data: return pd.DataFrame(columns=["id", "library_name", "topic", "created_at"])
                    return pd.DataFrame(data)[["id", "library_name", "topic", "created_at"]]

                def delete_doc(did):
                    try: session_manager.delete_documentation(int(did)); return f"✅ Doc {did} deleted.", get_docs()
                    except Exception as e: return f"❌ Error: {e}", get_docs()

                def add_doc(lib, topic, content):
                    try: session_manager.save_documentation(lib, topic, content); return "✅ Documentation saved.", get_docs()
                    except Exception as e: return f"❌ Error: {e}", get_docs()

                def seed_defaults():
                    defaults = [
                        {
                            "library": "Pandas",
                            "topic": "Handling Missing Data",
                            "content": """To handle missing data in Pandas:
1. Check: `df.isnull().sum()`
2. Drop: `df.dropna()` (careful)
3. Fill: `df.fillna(value)` or `df.interpolate()`
Always report missing counts before/after."""
                        },
                        {
                            "library": "Matplotlib",
                            "topic": "Agg Backend & Saving",
                            "content": """In this environment, Matplotlib must use the 'Agg' backend.
Standard Pattern:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

plt.figure(figsize=(10, 6))
# ... plotting code ...

buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
print('IMAGE_BASE64:' + base64.b64encode(buf.read()).decode())
plt.clf()
```"""
                        },
                        {
                            "library": "Seaborn",
                            "topic": "Styling",
                            "content": "Use `sns.set_theme(style='whitegrid')`. Palettes: 'viridis', 'Set2'."
                        }
                    ]
                    count = 0
                    for d in defaults:
                        session_manager.save_documentation(d["library"], d["topic"], d["content"])
                        count += 1
                    return f"✅ Added {count} default entries.", get_docs()

                d_file.upload(load_file_content, inputs=d_file, outputs=d_content)
                refresh_docs_btn.click(get_docs, outputs=docs_output)
                seed_defaults_btn.click(seed_defaults, outputs=[doc_msg, docs_output])
                delete_doc_btn.click(delete_doc, inputs=[doc_id_input], outputs=[doc_msg, docs_output])
                add_doc_btn.click(add_doc, inputs=[d_lib, d_topic, d_content], outputs=[doc_msg, docs_output])
                db_interface.load(get_docs, outputs=docs_output)

            # ================= PLANS TAB =================
            with gr.Tab("Planner Knowledge", id="tab_plans"):
                with gr.Row():
                    refresh_plans_btn = gr.Button("🔄 Refresh Plans")
                
                plans_output = gr.Dataframe(
                    headers=["id", "title", "content", "created_at"],
                    interactive=False,
                    wrap=True,
                    label="Saved Analysis Strategies"
                )
                
                with gr.Row():
                    plan_id_input = gr.Number(label="Plan ID", precision=0)
                    delete_plan_btn = gr.Button("🗑️ Delete Plan", variant="stop")
                
                plan_msg = gr.Markdown("")

                gr.Markdown("### ➕ Add Strategy Manually")
                with gr.Group():
                    pl_title = gr.Textbox(label="Title (e.g., 'Customer Churn Workflow')")
                    pl_content = gr.TextArea(label="Strategy Steps", lines=5)
                    add_plan_btn = gr.Button("💾 Save Strategy")

                def get_plans():
                    if not hasattr(session_manager, "get_all_analysis_plans"): return pd.DataFrame()
                    data = session_manager.get_all_analysis_plans()
                    return pd.DataFrame(data)[["id", "title", "content", "created_at"]] if data else pd.DataFrame(columns=["id", "title", "content", "created_at"])

                def delete_plan(pid):
                    try: session_manager.delete_analysis_plan(int(pid)); return f"✅ Plan {pid} deleted.", get_plans()
                    except Exception as e: return f"❌ Error: {e}", get_plans()

                def add_plan(t, c):
                    try: session_manager.save_analysis_plan(t, c); return "✅ Plan saved.", get_plans()
                    except Exception as e: return f"❌ Error: {e}", get_plans()

                refresh_plans_btn.click(get_plans, outputs=plans_output)
                delete_plan_btn.click(delete_plan, inputs=[plan_id_input], outputs=[plan_msg, plans_output])
                add_plan_btn.click(add_plan, inputs=[pl_title, pl_content], outputs=[plan_msg, plans_output])
                db_interface.load(get_plans, outputs=plans_output)

            # ================= CHAT HISTORY TAB =================
            with gr.Tab("Chat History", id="tab_chat"):
                with gr.Row():
                    hist_sess_id = gr.Textbox(label="Session ID", placeholder="Paste Session ID here...")
                    load_hist_btn = gr.Button("📜 Load Chat History")
                
                # Chatbot (Messages format - List of Dicts)
                history_chatbot = gr.Chatbot(label="Conversation History", height=600, elem_id="chat_history")
                artifacts_gallery = gr.Gallery(label="Session Artifacts (Plots/Files)", columns=4, height=200)
                
                def load_history(sid):
                    try:
                        print(f"DEBUG: [database_viewer.py] load_history called with sid='{sid}'", file=sys.stderr)
                        if not sid: 
                            print("DEBUG: [database_viewer.py] sid is empty, returning []", file=sys.stderr)
                            return [], []
                        
                        hist = session_manager.get_chat_history(sid)
                        print(f"DEBUG: [database_viewer.py] Retrieved hist type: {type(hist)}", file=sys.stderr)
                        if hist:
                            print(f"DEBUG: [database_viewer.py] First item type: {type(hist[0])}, Value: {hist[0]}", file=sys.stderr)

                        # Defensive check: Convert tuples to dicts if needed (Gradio 6.x compatibility)
                        if hist and isinstance(hist[0], (list, tuple)):
                            print("DEBUG: [database_viewer.py] Detected tuple format, converting to messages format", file=sys.stderr)
                            new_hist = []
                            for item in hist:
                                if len(item) >= 1 and item[0]: # User message
                                    new_hist.append({"role": "user", "content": str(item[0])})
                                if len(item) >= 2 and item[1]: # Assistant message
                                    new_hist.append({"role": "assistant", "content": str(item[1])})
                            hist = new_hist
                        
                        print(f"DEBUG: [database_viewer.py] Returning history (len={len(hist)}) to chatbot", file=sys.stderr)
                        return hist, session_manager.get_artifacts(sid)
                        
                    except Exception as e:
                        print(f"ERROR in load_history: {e}", file=sys.stderr)
                        return [], []
                
                load_hist_btn.click(load_history, inputs=[hist_sess_id], outputs=[history_chatbot, artifacts_gallery])

                # --- Event: Click Session Row -> Switch to Chat ---
                def on_select_session(evt: gr.SelectData, df):
                    if evt.index[1] is not None: # Clicked valid cell
                        try:
                            row_index = evt.index[0]
                            sid = df.iloc[row_index]['session_id']
                            hist = session_manager.get_chat_history(sid)
                            arts = session_manager.get_artifacts(sid)
                            # Return: SessionID, ChatHistory, Artifacts, TabID to switch to
                            return sid, hist, arts, "tab_chat"
                        except Exception as e:
                            print(f"Error selecting session: {e}")
                    return gr.update(), gr.update(), gr.update(), gr.update()

                sessions_output.select(
                    on_select_session,
                    inputs=[sessions_output],
                    outputs=[hist_sess_id, history_chatbot, artifacts_gallery, main_tabs]
                )

    return db_interface

if __name__ == "__main__":
    # Run this file directly to test the Database Viewer
    demo = create_database_interface()
    demo.launch()