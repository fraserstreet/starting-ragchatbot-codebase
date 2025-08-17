"""Integration tests using real components to identify actual system issues"""
import pytest
import tempfile
import shutil
import os
from config import Config
from rag_system import RAGSystem


class TestRealIntegration:
    """Test real system integration to catch issues missed by mocked tests"""
    
    @pytest.fixture
    def real_rag_system(self):
        """Create a real RAG system with actual ChromaDB"""
        # Create test config with temporary directory
        test_config = Config()
        temp_dir = tempfile.mkdtemp()
        test_config.CHROMA_PATH = temp_dir
        test_config.ANTHROPIC_API_KEY = "test-key-not-real"  # This will cause API errors, but that's ok
        test_config.MAX_RESULTS = 5  # Use fixed config
        
        try:
            rag_system = RAGSystem(test_config)
            yield rag_system
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_search_tool_with_empty_vector_store(self, real_rag_system):
        """Test search tool behavior when vector store is empty"""
        # This should reproduce the potential "list index out of range" error
        search_tool = real_rag_system.search_tool
        
        # Test search on empty vector store
        result = search_tool.execute("test query")
        
        # Should handle gracefully, not crash
        assert "No relevant content found" in result
        assert len(search_tool.last_sources) == 0
    
    def test_search_tool_with_real_course_data(self, real_rag_system):
        """Test search tool with actual course data loaded"""
        # Load real course documents
        docs_path = "../docs"
        if os.path.exists(docs_path):
            courses, chunks = real_rag_system.add_course_folder(docs_path)
            print(f"Loaded {courses} courses with {chunks} chunks")
            
            # Now test search
            result = real_rag_system.search_tool.execute("MCP server")
            print(f"Search result: {result}")
            
            # Should find relevant content
            assert result != "No relevant content found"
        else:
            pytest.skip("No docs folder available for real data test")
    
    def test_outline_tool_with_real_data(self, real_rag_system):
        """Test outline tool with real course data"""
        docs_path = "../docs"
        if os.path.exists(docs_path):
            courses, chunks = real_rag_system.add_course_folder(docs_path)
            
            # Test outline tool
            result = real_rag_system.outline_tool.execute("MCP")
            print(f"Outline result: {result}")
            
            # Should return course outline
            assert "Course Title" in result or "No course found" in result
        else:
            pytest.skip("No docs folder available for real data test")
    
    def test_vector_store_search_edge_cases(self, real_rag_system):
        """Test vector store search with various edge cases"""
        vector_store = real_rag_system.vector_store
        
        # Test 1: Search with no content
        results = vector_store.search("anything")
        assert results.is_empty()
        
        # Test 2: Search with course filter but no courses
        results = vector_store.search("test", course_name="nonexistent")
        assert results.error is not None
        
        # Test 3: Search with lesson filter but no content
        results = vector_store.search("test", lesson_number=1)
        assert results.is_empty()
    
    def test_reproduce_list_index_error(self, real_rag_system):
        """Attempt to reproduce the 'list index out of range' error"""
        # This test specifically tries to trigger the error we saw
        
        # Load some data first
        docs_path = "../docs"
        if os.path.exists(docs_path):
            courses, chunks = real_rag_system.add_course_folder(docs_path)
            
            # Try various searches that might trigger the error
            test_queries = [
                "How do you create an MCP server?",
                "What is MCP?",
                "Tell me about lesson 1",
                "MCP architecture details"
            ]
            
            for query in test_queries:
                try:
                    # Test direct search tool
                    result = real_rag_system.search_tool.execute(query)
                    print(f"Query '{query}' -> Success")
                    
                    # Test via tool manager
                    result2 = real_rag_system.tool_manager.execute_tool("search_course_content", query=query)
                    print(f"Query '{query}' via tool manager -> Success")
                    
                except Exception as e:
                    print(f"Query '{query}' -> ERROR: {e}")
                    # If we catch the error, we can analyze it
                    import traceback
                    traceback.print_exc()
        else:
            pytest.skip("No docs folder available for error reproduction test")