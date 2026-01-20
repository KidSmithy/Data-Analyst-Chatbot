import os
import asyncio
import sys
import gradio as gr
from database_viewer import create_database_interface
from ui import create_ui, set_research_agent, JS_SESSION_HANDLER
from agents.research_agent import ResearchAgent

# Windows Asyncio Policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    print("="*50)
    print("🚀 Launching Data Analyst App (Modular)")
    print("="*50)
    
    # Initialize and inject Research Agent
    print("🔄 Setting up Research Agent...")
    research_agent = ResearchAgent()
    set_research_agent(research_agent)
    
    # Create Blocks and capture components
    ui_blocks, ui_sidebar = create_ui()
    db_blocks = create_database_interface()
    
    with gr.Blocks(title="AI Workstation") as demo:
        # Inject JS globally since render() might lose the original load event
        demo.load(None, None, None, js=JS_SESSION_HANDLER)
        
        with gr.Tabs() as tabs:
            with gr.Tab("🤖 Data Analyst", id="tab_analyst") as t_analyst:
                ui_blocks.render()
            
            with gr.Tab("🗄️ Database Manager", id="tab_db") as t_db:
                db_blocks.render()
        
        # Event: Close sidebar when switching to Database Manager, Open when switching back
        t_db.select(lambda: gr.update(open=False), outputs=ui_sidebar)
        t_analyst.select(lambda: gr.update(open=True), outputs=ui_sidebar)
        
    demo.launch(server_name="127.0.0.1", server_port=7860)