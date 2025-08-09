# RAG System: Query to Response Flow

```
                    USER QUERY TO RESPONSE FLOW
                    ===========================

┌─────────────┐    1. POST /api/query     ┌─────────────┐
│   Frontend  │ ──────────────────────── ▶ │   FastAPI   │
│ (script.js) │   {query, session_id}     │   (app.py)  │
│   Browser   │                           │   Endpoint  │
└─────────────┘                           └─────────────┘
                                                   │
                                                   │ 2. rag_system.query()
                                                   ▼
                                          ┌─────────────┐
                                          │ RAG System  │
                                          │(rag_system  │
                                          │    .py)     │
                                          │Orchestrator │
                                          └─────────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
                    │ 3a. Get conversation        │ 3b. Generate response        │
                    │     history                  │     with tools               │
                    ▼                              ▼                              │
           ┌─────────────┐                ┌─────────────┐                         │
           │   Session   │                │ AI Generator│                         │
           │  Manager    │                │(ai_generator│                         │
           │(session_    │                │    .py)     │                         │
           │manager.py)  │                │Claude API + │                         │
           └─────────────┘                │   Tools     │                         │
                                          └─────────────┘                         │
                                                   │                              │
                                                   │ 4. Claude decides to        │
                                                   │    use search tool          │
                                                   ▼                              │
                                          ┌─────────────┐                         │
                                          │Tool Manager │                         │
                                          │(search_tools│                         │
                                          │    .py)     │                         │
                                          │CourseSearch │                         │
                                          │    Tool     │                         │
                                          └─────────────┘                         │
                                                   │                              │
                                                   │ 5. vector_store.search()    │
                                                   ▼                              │
                                          ┌─────────────┐                         │
                                          │Vector Store │                         │
                                          │(vector_store│                         │
                                          │    .py)     │                         │
                                          │ ChromaDB +  │                         │
                                          │ Embeddings  │                         │
                                          └─────────────┘                         │
                                                                                  │
                    RESPONSE FLOW (← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←)│
                                                                                  │
┌─────────────┐  10. Display response   ┌─────────────┐  9. JSON response      │
│   Frontend  │ ◀─────────────────────── │   FastAPI   │ ◀──────────────────────┘
│ (script.js) │   {answer, sources,      │   (app.py)  │   
│   Browser   │    session_id}           │   Endpoint  │   
└─────────────┘                         └─────────────┘   
                                                   ▲
                                                   │ 8. Return (response, sources)
                                                   │
                                          ┌─────────────┐
                                          │ RAG System  │ ◀─┐
                                          │(rag_system  │   │ 7. Generated response
                                          │    .py)     │   │    + source tracking
                                          │Orchestrator │   │
                                          └─────────────┘   │
                                                   ▲        │
                                                   │        │
                                          ┌─────────────┐   │
                                          │ AI Generator│ ──┘
                                          │(ai_generator│   
                                          │    .py)     │ ◀─┐
                                          │Claude API + │   │ 6. Search results
                                          │   Tools     │   │    as tool response
                                          └─────────────┘   │
                                                   ▲        │
                                                   │        │
                                          ┌─────────────┐   │
                                          │Tool Manager │ ──┘
                                          │(search_tools│   
                                          │    .py)     │ ◀─┐
                                          │CourseSearch │   │
                                          │    Tool     │   │
                                          └─────────────┘   │
                                                   ▲        │
                                                   │        │
                                          ┌─────────────┐   │
                                          │Vector Store │ ──┘
                                          │(vector_store│   
                                          │    .py)     │   
                                          │ ChromaDB +  │   
                                          │ Embeddings  │   
                                          └─────────────┘   

```

## Detailed Step-by-Step Flow:

### **Forward Flow (Query Processing):**

1. **Frontend → FastAPI**
   - User types query in `script.js`
   - `sendMessage()` creates POST to `/api/query`
   - Payload: `{query: "...", session_id: "..."}`

2. **FastAPI → RAG System** 
   - `app.py:query_documents()` receives request
   - Calls `rag_system.query(request.query, session_id)`

3. **RAG System Orchestration**
   - **3a**: Gets conversation history from `SessionManager`
   - **3b**: Calls `ai_generator.generate_response()` with tools

4. **AI Generator → Tool Manager**
   - Claude API receives query + available tools
   - Claude decides to use `search_course_content` tool
   - Calls `tool_manager.execute_tool()`

5. **Tool Manager → Vector Store**
   - `CourseSearchTool.execute()` runs
   - Calls `vector_store.search(query, course_name, lesson_number)`

6. **Vector Store Processing**
   - ChromaDB performs semantic search using embeddings
   - Returns `SearchResults` with documents, metadata, distances

### **Return Flow (Response Generation):**

6. **Vector Store → Tool Manager**
   - Search results formatted as tool response
   - Sources tracked in `tool.last_sources`

7. **Tool Manager → AI Generator**
   - Tool results sent back to Claude API
   - Claude generates natural language response using retrieved context

8. **AI Generator → RAG System**
   - Final response text returned
   - Sources extracted via `tool_manager.get_last_sources()`

9. **RAG System → FastAPI**
   - Updates conversation history via `session_manager.add_exchange()`
   - Returns `(response_text, sources_list)` tuple

10. **FastAPI → Frontend**
    - Wraps in `QueryResponse` JSON:
    ```json
    {
      "answer": "Generated response with context",
      "sources": ["Course Title Lesson X", ...],
      "session_id": "session_1"
    }
    ```

11. **Frontend Display**
    - `script.js` receives response
    - Converts markdown to HTML using `marked.parse()`
    - Displays with collapsible sources section

## Key Technical Details:

- **Session Management**: Maintains conversation context across queries
- **Tool Selection**: Claude autonomously decides when to search vs. use general knowledge  
- **Source Attribution**: Tracks which course chunks were used for each response
- **Error Handling**: Graceful fallbacks at each layer
- **Async Processing**: Non-blocking operations throughout pipeline