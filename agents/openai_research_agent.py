import os
import sys
import re
import json
import uuid
import asyncio
import inspect
import litellm
from typing import Optional, Any
from urllib.parse import urlencode
from contextlib import AsyncExitStack
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Google ADK & MCP Imports
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Import Database Manager for saving papers
# Add parent directory to path to ensure database module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from database import session_manager
except ImportError:
    session_manager = None

# --- OpenAI Configuration ---
# Fix typo in .env if present (OPENAI_APi_KEY -> OPENAI_API_KEY)
if "OPENAI_API_KEY" not in os.environ and "OPENAI_APi_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_APi_KEY"]

# Configuration
litellm.drop_params = True 
MODEL_NAME = "gpt-4.1" # Changed for stability and reliability
APP_NAME = "research_assistant_chatbot"
USER_ID = "user_123"

# Smithery/Provider Configuration
SEMANTIC_BASE_URL = "https://server.smithery.ai/@hamid-vakilzadeh/mcpsemanticscholar/mcp"
SEMANTIC_PARAMS = {
    "api_key": os.getenv("SMITHERY_API_KEY"), 
    "profile": "shrill-flamingo-rxCFEa" 
}
GOOGLE_SCHOLAR_BASE_URL = "https://server.smithery.ai/@mochow13/google-scholar-mcp/mcp"
GOOGLE_SCHOLAR_PARAMS = {
    "api_key": os.getenv("SMITHERY_API_KEY")
}

RESEARCH_AGENT_INSTRUCTION = """
You are an expert Academic Research Assistant. Your goal is to help users find, summarize, and analyze research papers.
You have access to external tools (Semantic Scholar, Google Scholar).

Guidelines:
1. Use Semantic Scholar or Google Scholar to fetch new data. ALWAYS set the limit to 5 results to prevent high costs and timeouts.
2. Provide Title, Year, Authors, and a full cohesive Abstract for people to understand the paper methodology and findings.
3. Synthesize information into a coherent answer.
"""

async def extract_papers_with_llm(text_content):
    """Uses LLM to parse unstructured text into structured paper JSON."""
    print("DEBUG: [ResearchAgent] 🧠 Using LLM to extract papers from text...", file=sys.stderr)
    prompt = f"""
    Extract research papers from the following text into a JSON list.
    Text: {text_content}
    
    Output format:
    [
      {{
        "title": "Paper Title",
        "url": "URL if available",
        "year": "Year",
        "authors": "Author names",
        "abstract": "Brief summary"
      }}
    ]
    Return ONLY valid JSON. Do not include markdown formatting like ```json.
    """
    try:
        response = await litellm.acompletion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
        # Clean markdown if present
        if "```" in content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match: content = match.group(1)
        return json.loads(content)
    except Exception as e:
        print(f"DEBUG: ⚠️ LLM Extraction failed: {e}", file=sys.stderr)
        return []

async def save_extracted_papers(tool_name, text_content, on_papers_found=None):
    """Parses tool output and saves paper metadata to the database if available."""
    if not session_manager or not hasattr(session_manager, "save_paper"):
        print("DEBUG: ⚠️ Database session_manager not connected. Cannot save papers.", file=sys.stderr)
        return

    print(f"DEBUG: [ResearchAgent] Processing tool output from {tool_name}...", file=sys.stderr)

    try:
        if not text_content or not text_content.strip():
            print("DEBUG: [ResearchAgent] Tool output is empty.", file=sys.stderr)
            return
            
        cleaned_content = text_content.strip()
        
        # Handle Markdown Code Blocks if present
        if "```" in cleaned_content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_content)
            if match:
                cleaned_content = match.group(1)

        try:
            data = json.loads(cleaned_content)
        except json.JSONDecodeError:
            # Fallback: Use LLM to extract papers from text
            data = await extract_papers_with_llm(cleaned_content)

        papers = []
        
        # Normalize data structure (Handle lists, dicts with 'data'/'results')
        if isinstance(data, list):
            papers = data
        elif isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list): papers = data["data"]
            elif "results" in data and isinstance(data["results"], list): papers = data["results"]
            else: papers = [data]

        print(f"DEBUG: [ResearchAgent] Extracted {len(papers)} potential items.", file=sys.stderr)

        for paper in papers:
            if not isinstance(paper, dict): continue
            
            # Map fields (Common fields for Semantic/Google Scholar)
            # Helper to get value case-insensitively
            def get_val(d, keys):
                for k in keys:
                    if k in d and d[k]: return d[k]
                return None

            url_val = get_val(paper, ["url", "link", "paperId", "externalIds"])
            if isinstance(url_val, (dict, list)): url_val = str(url_val)

            paper_info = {
                "source": tool_name,
                "title": get_val(paper, ["title", "Title", "name"]) or "Unknown Title",
                "url": url_val,
                "year": get_val(paper, ["year", "publication_date", "publicationDate", "date"]),
                "authors": get_val(paper, ["authors", "author"]),
                "abstract": get_val(paper, ["abstract", "snippet", "description"]),
                "raw_data": json.dumps(paper)
            }
            
            if paper_info["title"] and paper_info["title"] != "Unknown Title":
                print(f"DEBUG: [ResearchAgent] Found paper candidate: {paper_info['title']}", file=sys.stderr)
                session_manager.save_paper(paper_info)
                
                # Notify the agent instance to track this paper
                if on_papers_found:
                    on_papers_found(paper_info)
    except Exception as e:
        print(f"DEBUG: ⚠️ Error processing papers from {tool_name}: {e}", file=sys.stderr)

def create_tool_wrapper(session, tool_info, source_name, on_papers_found=None):
    """Dynamically creates a wrapper matching MCP schema for ADK."""
    async def wrapper(**kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        print(f"🛠️ [Research Tool] {tool_info.name} called with {list(clean_kwargs.keys())}")
        try:
            result = await session.call_tool(tool_info.name, clean_kwargs)
            text_content = ""
            if hasattr(result, 'content') and result.content:
                text_content = "\n".join([c.text for c in result.content if hasattr(c, 'text')])
            else:
                text_content = str(result)
            
            # Attempt to save extracted papers to DB
            asyncio.create_task(save_extracted_papers(source_name, text_content, on_papers_found))
            
            # Truncate to prevent ContextWindowExceeded or processing errors with large payloads
            return text_content[:25000] + ("... [Truncated]" if len(text_content) > 25000 else "")
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    safe_name = tool_info.name.replace("-", "_")
    wrapper.__name__ = safe_name
    wrapper.__doc__ = f"{tool_info.description}\nSource: {source_name}"

    # Patch Signature
    schema = tool_info.inputSchema
    props = schema.get("properties", {})
    required_params = schema.get("required", [])
    sig_params = []
    
    for param_name, param_info in props.items():
        json_type = param_info.get("type", "string")
        
        if json_type == "integer":
            annotation = int
        elif json_type == "boolean":
            annotation = bool
        elif json_type == "array":
            # OpenAI requires 'items' for arrays. Infer type or default to str.
            item_type = param_info.get("items", {}).get("type", "string")
            annotation = list[int] if item_type == "integer" else (list[bool] if item_type == "boolean" else list[str])
        else:
            annotation = str
            
        default = inspect.Parameter.empty
        if param_name not in required_params:
            default = None
            annotation = Optional[annotation]
        
        sig_params.append(inspect.Parameter(
            name=param_name, kind=inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation
        ))
    
    wrapper.__signature__ = inspect.Signature(parameters=sig_params)
    return wrapper

class ResearchAgent:
    def __init__(self):
        self.stack = AsyncExitStack()
        self.runner = None
        self.session_service = None
        self.initialized = False
        self.extra_tools = []
        self.found_papers = [] # Track papers found in current session

    def register_tool(self, tool_func):
        """Registers an external tool (function) to be used by the agent."""
        self.extra_tools.append(tool_func)

    async def initialize(self):
        """Connects to MCP servers and sets up the agent."""
        if self.initialized: return

        print("🔗 [Research Agent] Connecting to Scholar MCPs...")
        try:
            # Connect Semantic Scholar
            url_sem = f"{SEMANTIC_BASE_URL}?{urlencode(SEMANTIC_PARAMS)}"
            read_sem, write_sem, _ = await self.stack.enter_async_context(streamablehttp_client(url_sem))
            session_sem = await self.stack.enter_async_context(ClientSession(read_sem, write_sem))
            await session_sem.initialize()
            tools_sem = await session_sem.list_tools()
            
            # Connect Google Scholar
            url_goo = f"{GOOGLE_SCHOLAR_BASE_URL}?{urlencode(GOOGLE_SCHOLAR_PARAMS)}"
            read_goo, write_goo, _ = await self.stack.enter_async_context(streamablehttp_client(url_goo))
            session_goo = await self.stack.enter_async_context(ClientSession(read_goo, write_goo))
            await session_goo.initialize()
            tools_goo = await session_goo.list_tools()

            # Callback to track papers
            def on_paper(p): self.found_papers.append(p)

            # Wrap Tools
            adk_tools = []
            for t in tools_sem.tools:
                adk_tools.append(create_tool_wrapper(session_sem, t, "Semantic Scholar", on_paper))
            for t in tools_goo.tools:
                adk_tools.append(create_tool_wrapper(session_goo, t, "Google Scholar", on_paper))
            
            # Add dynamically registered tools (e.g. DB Search from UI)
            if self.extra_tools:
                adk_tools.extend(self.extra_tools)

            # Create Agent
            agent = LlmAgent(
                name="research_assistant",
                model=LiteLlm(model=MODEL_NAME),
                instruction=RESEARCH_AGENT_INSTRUCTION,
                tools=adk_tools
            )
            
            self.session_service = InMemorySessionService()
            self.runner = Runner(agent=agent, app_name=APP_NAME, session_service=self.session_service)
            self.initialized = True
            print(f"✅ [Research Agent] Ready with {len(adk_tools)} tools.")
            
        except Exception as e:
            print(f"❌ [Research Agent] Initialization Failed: {e}")
            raise

    async def process_message(self, message: str, session_id: str):
        """Processes a user message."""
        if not self.initialized:
            await self.initialize()

        # Reset found papers for this turn
        self.found_papers = []

        # Ensure session exists
        s = await self.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        if s is None:
            s = await self.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

        # --- RAG LOGIC START ---
        local_context = ""
        if session_manager:
            print(f"DEBUG: [ResearchAgent] 🔍 Checking local library (RAG) for: {message}", file=sys.stderr)
            try:
                # Perform vector/keyword search (Run in thread to avoid blocking async loop)
                search_result_json = await asyncio.to_thread(session_manager.ai_search_papers, message)
                search_results = json.loads(search_result_json)
                
                # Check if we got a list of papers (success) or a dict (no_results)
                if isinstance(search_results, list) and len(search_results) > 0:
                    print(f"DEBUG: [ResearchAgent] ✅ Found {len(search_results)} local papers.", file=sys.stderr)
                    
                    # 1. Add to found_papers so they appear in the UI
                    for p in search_results:
                        self.found_papers.append(p)
                    
                    # 2. Construct Context for LLM
                    local_context = (
                        f"=== LOCAL LIBRARY RESULTS ===\n"
                        f"{json.dumps(search_results, indent=2)}\n"
                        f"=============================\n\n"
                    )
                else:
                    print("DEBUG: [ResearchAgent] ❌ No relevant local papers found.", file=sys.stderr)
            except Exception as e:
                print(f"DEBUG: [ResearchAgent] ⚠️ RAG Search Error: {e}", file=sys.stderr)
        else:
            print("DEBUG: [ResearchAgent] ⚠️ Session Manager not available. Skipping RAG.", file=sys.stderr)
        
        if local_context:
            prompt_text = f"{local_context}User Query: {message}\n\nINSTRUCTIONS:\n1. Relevant papers were found in the local library (see above).\n2. Answer the user's query using ONLY the provided local papers if they are sufficient.\n3. If the local papers are NOT sufficient, ONLY THEN use external tools to find new information."
        else:
            prompt_text = message
        # --- RAG LOGIC END ---

        input_msg = types.Content(role='user', parts=[types.Part(text=prompt_text)])
        final_response = "I couldn't find any information."

        try:
            async for event in self.runner.run_async(user_id=USER_ID, session_id=s.id, new_message=input_msg):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text
            
            # Return structured response so UI can trigger Deep Dive Agent
            result = {"text": final_response, "papers": self.found_papers}
            print(f"DEBUG: [ResearchAgent] process_message returning type: {type(result)} with keys: {list(result.keys())}", file=sys.stderr)
            return result
        except Exception as e:
            return f"Error during research: {str(e)}"

    async def deep_dive_into_papers(self, papers, session_id):
        """
        PARALLEL EXECUTION: Spawns sub-tasks to fetch details for multiple papers simultaneously.
        This implements the 'Parallel Agent' pattern for the deep dive phase.
        """
        if not papers: return
        
        # Limit deep dive to top 5 papers to save costs
        papers_to_process = papers[:5]
        print(f"DEBUG: 🕵️ Sequential Follow-up Agent (Deep Dive) started for {len(papers_to_process)} papers (capped at 5)...", file=sys.stderr)
        # Use a distinct session ID prefix to keep deep dive history separate from main chat
        deep_dive_session = f"deep_dive_{session_id}"
        
        tasks = []
        for paper in papers_to_process:
            tasks.append(self._process_single_paper(paper, deep_dive_session))
            
        # Run all deep dives in parallel
        await asyncio.gather(*tasks)

    async def _process_single_paper(self, paper, session_id):
        try:
            title = paper.get("title")
            url = paper.get("url")
            if not title or not url: return

            print(f"DEBUG: 🕵️ Deep diving into: {title}", file=sys.stderr)
            deep_query = f"Provide a detailed abstract, key findings, and methodology for the paper titled '{title}'. If available, include the publication date and authors."
            
            # Call the agent itself to perform the research
            response_data = await self.process_message(deep_query, session_id)
            
            content = response_data.get("text", "") if isinstance(response_data, dict) else str(response_data)

            if session_manager:
                # This populates the database with the deep dive results
                session_manager.update_paper(url, abstract=content)
                print(f"DEBUG: ✅ Deep Dive updated details for {title}", file=sys.stderr)
        except Exception as e:
            print(f"DEBUG: ❌ Deep dive failed for {title}: {e}", file=sys.stderr)

    async def shutdown(self):
        await self.stack.aclose()