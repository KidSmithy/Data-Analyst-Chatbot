import sqlite3
import sys
import uuid
import json
import litellm
import os
import sqlite_vec
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "data_analyst_v3.db"

class SessionManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.row_factory = sqlite3.Row
        self._init_database()
    
    def _init_database(self):
        cursor = self.conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                file_name TEXT,
                file_type TEXT,
                data_preview TEXT,
                data_summary TEXT
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')
        
        # Findings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                finding_type TEXT,
                content TEXT,
                tags TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')
        
        # Artifacts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                artifact_type TEXT,
                file_name TEXT,
                file_path TEXT,
                metadata TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')
        
        # Papers table (For RAG/Local Library)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                title TEXT,
                url TEXT UNIQUE,
                year TEXT,
                authors TEXT,
                abstract TEXT,
                raw_data TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        # Migration: Add embedding column if it doesn't exist
        try:
            cursor.execute("SELECT embedding FROM papers LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE papers ADD COLUMN embedding TEXT")
            self.conn.commit()
            
        # Create Vector Table (sqlite-vec)
        # We use float[1536] for text-embedding-3-small
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_papers USING vec0(
                embedding float[1536]
            )
        ''')
        
        # Migration: Populate vec_papers if empty but papers has data
        cursor.execute("SELECT count(*) FROM vec_papers")
        if cursor.fetchone()[0] == 0:
            print("DEBUG: [database.py] Migrating embeddings to sqlite-vec...", file=sys.stderr)
            cursor.execute("SELECT id, embedding FROM papers WHERE embedding IS NOT NULL")
            for row in cursor.fetchall():
                if row['embedding']:
                    try:
                        emb = json.loads(row['embedding'])
                        cursor.execute("INSERT INTO vec_papers(rowid, embedding) VALUES (?, ?)", (row['id'], emb))
                    except Exception: pass
            self.conn.commit()

        self.conn.commit()
    
    def create_session(self, file_path: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        file_name = Path(file_path).name if file_path else None
        file_type = Path(file_path).suffix.lower() if file_path else None
        
        title = f"Analysis of {file_name}" if file_path else "New Analysis Session"
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id, title, created_at, updated_at, 
                                 file_name, file_type, data_preview, data_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, title, datetime.now(), datetime.now(),
              file_name, file_type, None, None))
        
        self.conn.commit()
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def save_message(self, session_id: str, role: str, content: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (session_id, role, content, datetime.now()))
        cursor.execute('UPDATE sessions SET updated_at = ? WHERE session_id = ?', 
                      (datetime.now(), session_id))
        self.conn.commit()
    
    def get_messages(self, session_id: str, limit: int = 20) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT role, content, timestamp FROM messages 
            WHERE session_id = ? ORDER BY timestamp LIMIT ?
        ''', (session_id, limit))
        return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in cursor.fetchall()]

    def save_finding(self, session_id: str, finding_type: str, content: str, tags: Optional[str] = None) -> None:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO findings (session_id, finding_type, content, tags, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, finding_type, content, tags, datetime.now()))
        self.conn.commit()
    
    def search_findings(self, query: str, limit: int = 5) -> List[Dict]:
        cursor = self.conn.cursor()
        search_term = f"%{query}%"
        cursor.execute('''
            SELECT s.session_id, s.title, f.content, f.tags, f.created_at
            FROM findings f JOIN sessions s ON f.session_id = s.session_id
            WHERE f.content LIKE ? OR f.tags LIKE ?
            ORDER BY f.created_at DESC LIMIT ?
        ''', (search_term, search_term, limit))
        return [dict(row) for row in cursor.fetchall()]

    def save_artifact(self, session_id: str, artifact_type: str, file_name: str, file_path: str, metadata: Optional[Dict] = None) -> None:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO artifacts (session_id, artifact_type, file_name, file_path, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, artifact_type, file_name, file_path, json.dumps(metadata) if metadata else None, datetime.now()))
        self.conn.commit()
    
    def update_session_data(self, session_id: str, data_preview: str, data_summary: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE sessions SET data_preview = ?, data_summary = ?, updated_at = ?
            WHERE session_id = ?
        ''', (data_preview, data_summary, datetime.now(), session_id))
        self.conn.commit()

    def update_session_title(self, session_id: str, title: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute('UPDATE sessions SET title = ? WHERE session_id = ?', (title, session_id))
        self.conn.commit()

    def list_sessions(self, limit: int = 10) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT session_id, title, file_name, created_at, updated_at
            FROM sessions ORDER BY updated_at DESC LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates vector embedding for text using LiteLLM."""
        try:
            if not text or len(text) < 5: 
                print(f"DEBUG: [database.py] Text too short for embedding.", file=sys.stderr)
                return None
            # Use text-embedding-3-small (OpenAI) or similar. 
            # Ensure OPENAI_API_KEY is set in .env
            if not os.getenv("OPENAI_API_KEY"): 
                print("DEBUG: [database.py] OPENAI_API_KEY missing. Cannot generate embeddings.", file=sys.stderr)
                return None
            
            response = litellm.embedding(model="text-embedding-3-small", input=[text])
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"⚠️ Embedding generation failed: {e}", file=sys.stderr)
            return None

    def save_paper(self, paper_info: Dict) -> None:
        """Saves a research paper to the local library."""
        cursor = self.conn.cursor()
        try:
            # Convert list fields to string for storage
            authors = str(paper_info.get("authors")) if isinstance(paper_info.get("authors"), list) else paper_info.get("authors")
            
            # Ensure URL is unique (fix for manual entries)
            url = paper_info.get("url")
            if not url:
                url = f"manual_{uuid.uuid4()}"
            
            # Generate Embedding
            content = f"{paper_info.get('title', '')} {paper_info.get('abstract', '')}"
            embedding = self._get_embedding(content)
            embedding_json = json.dumps(embedding) if embedding else None
            
            cursor.execute('''
                INSERT OR IGNORE INTO papers (source, title, url, year, authors, abstract, raw_data, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper_info.get("source"),
                paper_info.get("title"),
                url,
                paper_info.get("year"),
                authors,
                paper_info.get("abstract"),
                paper_info.get("raw_data"),
                embedding_json,
                datetime.now()
            ))
            
            if cursor.rowcount > 0:
                paper_id = cursor.lastrowid
                # Save to Vector Store
                if embedding:
                    cursor.execute("INSERT INTO vec_papers(rowid, embedding) VALUES (?, ?)", (paper_id, embedding))
                
                print(f"DEBUG: [database.py] 📄 Paper saved: {paper_info.get('title', 'Unknown')}", file=sys.stderr)
            
            self.conn.commit()
        except Exception as e:
            print(f"Error saving paper: {e}", file=sys.stderr)

    def update_paper(self, url: str, abstract: str = None, raw_data: str = None) -> None:
        """Updates an existing paper's details (used by Deep Dive Agent)."""
        cursor = self.conn.cursor()
        try:
            if abstract:
                cursor.execute('UPDATE papers SET abstract = ? WHERE url = ?', (abstract, url))
                
                # --- SYNC EMBEDDING (Critical for RAG) ---
                # Fetch ID and Title to regenerate embedding with new abstract
                cursor.execute('SELECT id, title FROM papers WHERE url = ?', (url,))
                row = cursor.fetchone()
                if row:
                    pid, title = row
                    content = f"{title} {abstract}"
                    embedding = self._get_embedding(content)
                    
                    # Always remove old vector first to prevent stale data
                    cursor.execute("DELETE FROM vec_papers WHERE rowid = ?", (pid,))
                    
                    if embedding:
                        cursor.execute("INSERT INTO vec_papers(rowid, embedding) VALUES (?, ?)", (pid, embedding))
                        # Keep text backup in sync
                        cursor.execute("UPDATE papers SET embedding = ? WHERE id = ?", (json.dumps(embedding), pid))
                    else:
                        cursor.execute("UPDATE papers SET embedding = NULL WHERE id = ?", (pid,))

            if raw_data:
                cursor.execute('UPDATE papers SET raw_data = ? WHERE url = ?', (raw_data, url))
            self.conn.commit()
        except Exception as e:
            print(f"Error updating paper: {e}", file=sys.stderr)

    def update_paper_by_id(self, paper_id: int, title: str, year: str, authors: str, source: str, abstract: str) -> None:
        """Updates paper details manually via the UI."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE papers 
            SET title = ?, year = ?, authors = ?, source = ?, abstract = ?
            WHERE id = ?
        ''', (title, year, authors, source, abstract, paper_id))
        
        # --- SYNC EMBEDDING ---
        try:
            content = f"{title} {abstract}"
            embedding = self._get_embedding(content)
            
            # Always remove old vector first to prevent stale data
            cursor.execute("DELETE FROM vec_papers WHERE rowid = ?", (paper_id,))
            
            if embedding:
                cursor.execute("INSERT INTO vec_papers(rowid, embedding) VALUES (?, ?)", (paper_id, embedding))
                cursor.execute("UPDATE papers SET embedding = ? WHERE id = ?", (json.dumps(embedding), paper_id))
            else:
                cursor.execute("UPDATE papers SET embedding = NULL WHERE id = ?", (paper_id,))
        except Exception as e:
            print(f"Error updating embedding for paper {paper_id}: {e}", file=sys.stderr)
            
        self.conn.commit()

    def search_papers(self, query: str, limit: int = 5) -> List[Dict]:
        """Searches local papers by title or abstract."""
        cursor = self.conn.cursor()
        search_term = f"%{query}%"
        cursor.execute('''
            SELECT title, year, authors, abstract, url 
            FROM papers 
            WHERE title LIKE ? OR abstract LIKE ? 
            ORDER BY created_at DESC LIMIT ?
        ''', (search_term, search_term, limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def ai_search_papers(self, query: str) -> str:
        """
        Tool for AI Agent: Searches local database for papers matching the query using Vector Search (Semantic) + Keyword Fallback.
        Returns: JSON string of found papers.
        """
        print(f"DEBUG: [database.py] AI Tool searching for: '{query}'", file=sys.stderr)
        cursor = self.conn.cursor()
        
        # 1. Try Vector Search first
        query_embedding = self._get_embedding(query)
        results = []
        
        if query_embedding:
            try:
                # sqlite-vec Search
                # Note: distance is usually L2 or Cosine distance. 
                # For normalized vectors (OpenAI), L2 distance sorts same as Cosine Similarity.
                cursor.execute('''
                    SELECT rowid, distance 
                    FROM vec_papers 
                    WHERE embedding MATCH ? 
                    AND k = 5 
                    ORDER BY distance
                ''', (query_embedding,))
                
                vec_rows = cursor.fetchall()
                
                results = []
                for vr in vec_rows:
                    # Fetch full paper details
                    cursor.execute("SELECT title, year, authors, abstract, url FROM papers WHERE id = ?", (vr['rowid'],))
                    paper = cursor.fetchone()
                    if paper:
                        results.append(dict(paper))
                
                print(f"DEBUG: [database.py] Vector search found {len(results)} results.", file=sys.stderr)
                    
            except Exception as e:
                print(f"⚠️ Vector search error: {e}", file=sys.stderr)
        else:
            print("DEBUG: [database.py] ⚠️ Query embedding could not be generated (check API key or text length).", file=sys.stderr)
        
        # 2. Fallback to Keyword Search if no vector results
        if not results:
            print("DEBUG: [database.py] Falling back to Keyword Search", file=sys.stderr)
            search_term = f"%{query.strip()}%"
            print(f"DEBUG: [database.py] Executing SQL LIKE with term: '{search_term}'", file=sys.stderr)
            
            # Added 'authors' to the search columns
            sql = "SELECT title, year, authors, abstract, url FROM papers WHERE title LIKE ? OR abstract LIKE ? OR authors LIKE ? ORDER BY created_at DESC LIMIT 5"
            cursor.execute(sql, (search_term, search_term, search_term))
            results = [dict(row) for row in cursor.fetchall()]
            print(f"DEBUG: [database.py] Keyword search found {len(results)} results.", file=sys.stderr)
        
        if not results:
            return json.dumps({"status": "no_results", "message": "No local papers found matching query."})
            
        return json.dumps(results, indent=2)

    def get_all_sessions(self) -> List[Dict]:
        """Retrieves all sessions for management."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM sessions ORDER BY updated_at DESC')
        return [dict(row) for row in cursor.fetchall()]

    def delete_session(self, session_id: str) -> None:
        """Deletes a session and all associated data."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM findings WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM artifacts WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        self.conn.commit()

    def get_all_findings(self) -> List[Dict]:
        """Retrieves all findings with session details."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT f.*, s.title as session_title 
            FROM findings f 
            LEFT JOIN sessions s ON f.session_id = s.session_id 
            ORDER BY f.created_at DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]

    def delete_finding(self, finding_id: int) -> None:
        """Deletes a finding by ID."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM findings WHERE id = ?', (finding_id,))
        self.conn.commit()

    def get_all_papers(self) -> List[Dict]:
        """Retrieves all papers."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM papers ORDER BY created_at DESC')
        return [dict(row) for row in cursor.fetchall()]

    def delete_paper(self, paper_id: int) -> None:
        """Deletes a paper by ID."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM papers WHERE id = ?', (paper_id,))
        cursor.execute('DELETE FROM vec_papers WHERE rowid = ?', (paper_id,))
        self.conn.commit()

    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Retrieves chat history formatted for Gradio Chatbot (list of dicts)."""
        print(f"DEBUG: [database.py] Fetching chat history for session '{session_id}'", file=sys.stderr)
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT role, content FROM messages 
            WHERE session_id = ? ORDER BY timestamp ASC
        ''', (session_id,))
        
        rows = cursor.fetchall()
        # Format: [{"role": "user", "content": "msg"}, {"role": "assistant", "content": "msg"}]
        history = []
        for row in rows:
            role = row[0] if row[0] else "user"
            content = row[1] if row[1] is not None else ""
            # Ensure strict format for Gradio 6.x (lowercase role, string content)
            history.append({"role": str(role).lower(), "content": str(content)})
            
        if history:
            print(f"DEBUG: [database.py] History format sample: {history[0]}", file=sys.stderr)
        else:
            print(f"DEBUG: [database.py] History is empty for session '{session_id}'", file=sys.stderr)
        return history

    def get_artifacts(self, session_id: str) -> List[str]:
        """Retrieves artifact file paths for a session."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT file_path FROM artifacts WHERE session_id = ? ORDER BY created_at ASC', (session_id,))
        return [row[0] for row in cursor.fetchall()]

    def reset_database(self) -> None:
        """Deletes all sessions, messages, findings, and artifacts."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM messages')
        cursor.execute('DELETE FROM findings')
        cursor.execute('DELETE FROM artifacts')
        cursor.execute('DELETE FROM sessions')
        self.conn.commit()

# Global Instance
session_manager = SessionManager()