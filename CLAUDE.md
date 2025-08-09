# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
uv sync

# Create environment file (copy from template)
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Development Workflow
```bash
# Run with auto-reload during development
cd backend && uv run uvicorn app:app --reload --port 8000

# Add new course documents
# Place .txt files in docs/ directory - they will be auto-processed on startup
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials built with FastAPI backend and vanilla JavaScript frontend.

### Core Architecture Pattern
The system follows a **tool-based RAG architecture** where:
1. User queries are processed by Claude via Anthropic API
2. Claude autonomously decides when to use semantic search tools
3. Vector search retrieves relevant course chunks from ChromaDB  
4. Claude synthesizes responses using retrieved context
5. Conversation history is maintained across sessions

### Key Components

**Backend (Python/FastAPI):**
- `rag_system.py` - **Main orchestrator** that coordinates all components
- `vector_store.py` - ChromaDB integration with dual collections (metadata + content)
- `document_processor.py` - Structured parsing of course documents with lesson segmentation
- `ai_generator.py` - Anthropic Claude API client with tool execution capability
- `search_tools.py` - Tool definitions for Claude's function calling
- `session_manager.py` - Conversation history management
- `app.py` - FastAPI server with `/api/query` and `/api/courses` endpoints

**Frontend (Vanilla JS):**
- `script.js` - Chat interface with API communication and markdown rendering
- `index.html` - Single-page application with sidebar for course stats
- `style.css` - UI styling

### Data Flow Architecture

**Document Processing (Startup):**
```
Course Files (.txt) ’ DocumentProcessor ’ Vector Embeddings ’ ChromaDB Collections
                                      “
                              [course_catalog] + [course_content]
```

**Query Processing (Runtime):**
```
User Query ’ FastAPI ’ RAG System ’ Claude API + Tools ’ Vector Search ’ Response
```

### Document Structure Requirements
Course documents must follow this format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [lesson title]
Lesson Link: [lesson url]
[lesson content]

Lesson 1: [next lesson title]
...
```

### Configuration System
All settings centralized in `backend/config.py`:
- `CHUNK_SIZE: 800` - Text chunk size for embeddings
- `CHUNK_OVERLAP: 100` - Overlap between chunks for context preservation
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"` - Sentence transformer model
- `MAX_RESULTS: 5` - Vector search result limit
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"` - Claude model version

### Vector Storage Design
**Dual ChromaDB Collections:**
- `course_catalog` - Course metadata for semantic course name resolution
- `course_content` - Text chunks with course/lesson attribution for content search

**Chunk Enhancement Strategy:**
- First chunk per lesson: `"Lesson X content: [text]"`
- Other chunks: `"Course [title] Lesson X content: [text]"`

### Tool Architecture
The system uses **Anthropic's tool calling** where Claude decides when to search:
- `CourseSearchTool` - Semantic search with course/lesson filtering
- Tool results automatically tracked for source attribution
- Claude generates responses by synthesizing search results

### Session Management
- Conversation history maintained per session (configurable limit)
- Session IDs generated automatically if not provided
- History passed to Claude for contextual responses

## Working with This Codebase

### Adding New Course Documents
1. Place `.txt` files in `docs/` directory following the required format
2. Restart server - documents are automatically processed and indexed
3. Verify via `/api/courses` endpoint or frontend sidebar

### Modifying Search Behavior
- Adjust search parameters in `backend/config.py`
- Modify tool definitions in `search_tools.py`
- Update AI system prompt in `ai_generator.py`

### Extending Functionality
- Add new tools by extending `Tool` class in `search_tools.py`
- Register tools in `RAGSystem.__init__()` in `rag_system.py`
- New API endpoints go in `app.py`

### Frontend Customization
- Chat interface logic in `script.js:sendMessage()`
- Markdown rendering uses marked.js library
- UI styling entirely in `style.css`

## Key Technical Decisions

**Tool-Based vs. Traditional RAG:** This system lets Claude decide when to search rather than always retrieving, making responses more natural for general questions.

**Dual Vector Collections:** Separates course metadata from content for efficient course name resolution and content search.

**Sentence-Based Chunking:** Preserves semantic boundaries while maintaining overlap for context.

**Context Enhancement:** Prefixes chunks with course/lesson information to improve retrieval relevance.

**Standard I/O vs. HTTP:** Uses FastAPI for web interface rather than MCP standard I/O transport pattern.
- always use uv to run the server do not use pip directly