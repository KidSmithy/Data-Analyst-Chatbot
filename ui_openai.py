import gradio as gr
import sys
import asyncio
from pathlib import Path
from config import GLOBAL_MEMORY
from database import session_manager

# --- 1. MONKEY PATCH: Force 'agents.data_agent' to be 'openai_data_agent' ---
# This ensures that when ui.py imports 'agents.data_agent', it gets our OpenAI version.
from agents import openai_data_agent
sys.modules["agents.data_agent"] = openai_data_agent

# --- 2. Import Research Agent ---
from agents.openai_research_agent import ResearchAgent

# --- 3. Import UI Components ---
from ui import create_ui, set_research_agent

if __name__ == "__main__":
    print("🚀 Launching OpenAI-Powered AI Workstation...")
    
    # Initialize the OpenAI Research Agent
    print("🔄 Initializing OpenAI Research Agent...")
    research_agent = ResearchAgent()
    
    # Inject it into the UI module so bot_respond uses it
    set_research_agent(research_agent)
    
    # Launch UI
    demo, sidebar = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7861) # Use a different port (7861)