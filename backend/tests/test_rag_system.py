import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test RAG system integration"""
    
    @pytest.fixture
    def temp_rag_system(self, test_config, mock_embedding_function):
        """Create a temporary RAG system for testing"""
        temp_dir = tempfile.mkdtemp()
        test_config.CHROMA_PATH = temp_dir
        
        try:
            with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_ef:
                mock_ef.return_value = mock_embedding_function
                
                with patch('anthropic.Anthropic') as mock_anthropic:
                    mock_client = Mock()
                    mock_anthropic.return_value = mock_client
                    
                    rag_system = RAGSystem(test_config)
                    yield rag_system, mock_client
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def broken_rag_system(self, broken_config, mock_embedding_function):
        """Create RAG system with broken config (MAX_RESULTS=0)"""
        temp_dir = tempfile.mkdtemp()
        broken_config.CHROMA_PATH = temp_dir
        
        try:
            with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_ef:
                mock_ef.return_value = mock_embedding_function
                
                with patch('anthropic.Anthropic') as mock_anthropic:
                    mock_client = Mock()
                    mock_anthropic.return_value = mock_client
                    
                    rag_system = RAGSystem(broken_config)
                    yield rag_system, mock_client
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_init_components(self, temp_rag_system):
        """Test that RAG system initializes all components correctly"""
        rag_system, _ = temp_rag_system
        
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None
        
        # Verify tools are registered
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        tool_names = [td["name"] for td in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
    
    def test_add_course_document_success(self, temp_rag_system, sample_course):
        """Test successful course document addition"""
        rag_system, _ = temp_rag_system
        
        # Mock document processor
        sample_chunks = [
            CourseChunk(content="Sample content", course_title=sample_course.title, lesson_number=1, chunk_index=0)
        ]
        rag_system.document_processor.process_course_document = Mock(return_value=(sample_course, sample_chunks))
        
        # Mock vector store methods
        rag_system.vector_store.add_course_metadata = Mock()
        rag_system.vector_store.add_course_content = Mock()
        
        course, chunk_count = rag_system.add_course_document("/fake/path.txt")
        
        # Verify processing was called
        rag_system.document_processor.process_course_document.assert_called_once_with("/fake/path.txt")
        
        # Verify vector store was updated
        rag_system.vector_store.add_course_metadata.assert_called_once_with(sample_course)
        rag_system.vector_store.add_course_content.assert_called_once_with(sample_chunks)
        
        assert course == sample_course
        assert chunk_count == 1
    
    def test_add_course_document_failure(self, temp_rag_system):
        """Test handling of course document processing failure"""
        rag_system, _ = temp_rag_system
        
        # Mock document processor to raise exception
        rag_system.document_processor.process_course_document = Mock(side_effect=Exception("Processing failed"))
        
        course, chunk_count = rag_system.add_course_document("/fake/path.txt")
        
        assert course is None
        assert chunk_count == 0
    
    def test_add_course_folder_success(self, temp_rag_system, sample_course):
        """Test successful course folder processing"""
        rag_system, _ = temp_rag_system
        
        # Create temporary folder with test files
        temp_folder = tempfile.mkdtemp()
        try:
            # Create test files
            test_file1 = os.path.join(temp_folder, "course1.txt")
            test_file2 = os.path.join(temp_folder, "course2.pdf")
            non_course_file = os.path.join(temp_folder, "readme.md")
            
            with open(test_file1, 'w') as f:
                f.write("Course content")
            with open(test_file2, 'w') as f:
                f.write("PDF course content")
            with open(non_course_file, 'w') as f:
                f.write("Not a course file")
            
            # Mock vector store methods
            rag_system.vector_store.get_existing_course_titles = Mock(return_value=[])
            rag_system.vector_store.add_course_metadata = Mock()
            rag_system.vector_store.add_course_content = Mock()
            
            # Mock document processor to return different courses for different files
            def mock_process_document(file_path):
                if "course1.txt" in file_path:
                    course1 = Course(title="Course 1", instructor="Instructor 1", lessons=[])
                    chunks1 = [CourseChunk(content="Content", course_title="Course 1", lesson_number=1, chunk_index=0)]
                    return course1, chunks1
                else:  # course2.pdf
                    course2 = Course(title="Course 2", instructor="Instructor 2", lessons=[])
                    chunks2 = [CourseChunk(content="Content", course_title="Course 2", lesson_number=1, chunk_index=0)]
                    return course2, chunks2
            
            rag_system.document_processor.process_course_document = Mock(side_effect=mock_process_document)
            
            total_courses, total_chunks = rag_system.add_course_folder(temp_folder)
            
            # Should process 2 course files (.txt and .pdf), ignore .md
            assert rag_system.document_processor.process_course_document.call_count == 2
            assert total_courses == 2
            assert total_chunks == 2
            
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)
    
    def test_add_course_folder_with_existing_courses(self, temp_rag_system, sample_course):
        """Test course folder processing with existing courses"""
        rag_system, _ = temp_rag_system
        
        temp_folder = tempfile.mkdtemp()
        try:
            test_file = os.path.join(temp_folder, "course.txt")
            with open(test_file, 'w') as f:
                f.write("Course content")
            
            # Mock existing course
            rag_system.vector_store.get_existing_course_titles = Mock(return_value=[sample_course.title])
            
            # Mock document processor
            sample_chunks = [CourseChunk(content="Content", course_title=sample_course.title, lesson_number=1, chunk_index=0)]
            rag_system.document_processor.process_course_document = Mock(return_value=(sample_course, sample_chunks))
            
            total_courses, total_chunks = rag_system.add_course_folder(temp_folder)
            
            # Should not add existing course
            assert total_courses == 0
            assert total_chunks == 0
            
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)
    
    def test_add_course_folder_clear_existing(self, temp_rag_system):
        """Test course folder processing with clear_existing=True"""
        rag_system, _ = temp_rag_system
        
        # Mock vector store
        rag_system.vector_store.clear_all_data = Mock()
        
        temp_folder = tempfile.mkdtemp()
        try:
            total_courses, total_chunks = rag_system.add_course_folder(temp_folder, clear_existing=True)
            
            # Verify data was cleared
            rag_system.vector_store.clear_all_data.assert_called_once()
            
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)
    
    def test_add_course_folder_nonexistent(self, temp_rag_system):
        """Test course folder processing with nonexistent folder"""
        rag_system, _ = temp_rag_system
        
        total_courses, total_chunks = rag_system.add_course_folder("/nonexistent/folder")
        
        assert total_courses == 0
        assert total_chunks == 0
    
    def test_query_without_session(self, temp_rag_system):
        """Test query processing without session ID"""
        rag_system, mock_client = temp_rag_system
        
        # Mock AI response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a response about machine learning")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        response, sources = rag_system.query("What is machine learning?")
        
        # Verify AI was called
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert "What is machine learning?" in call_args["messages"][0]["content"]
        assert "tools" in call_args  # Tools should be available
        
        assert response == "This is a response about machine learning"
        assert isinstance(sources, list)
    
    def test_query_with_session(self, temp_rag_system):
        """Test query processing with session ID"""
        rag_system, mock_client = temp_rag_system
        
        # Mock AI response
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with context")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        # Mock session manager
        rag_system.session_manager.get_conversation_history = Mock(return_value="Previous conversation")
        rag_system.session_manager.add_exchange = Mock()
        
        response, sources = rag_system.query("Follow up question", session_id="test_session")
        
        # Verify session methods were called
        rag_system.session_manager.get_conversation_history.assert_called_once_with("test_session")
        rag_system.session_manager.add_exchange.assert_called_once_with(
            "test_session", "Follow up question", "Response with context"
        )
        
        # Verify AI was called with history
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation" in call_args["system"]
    
    def test_query_with_tool_usage_working_config(self, temp_rag_system):
        """Test query that triggers tool usage with working config"""
        rag_system, mock_client = temp_rag_system
        
        # Mock tool use response
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool1"
        tool_block.input = {"query": "machine learning"}
        
        initial_response = Mock()
        initial_response.content = [tool_block]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Based on search results: ML is...")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock vector store search
        from vector_store import SearchResults
        rag_system.vector_store.search = Mock(return_value=SearchResults(
            documents=["Machine learning is a subset of AI"],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1]
        ))
        rag_system.vector_store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
        
        response, sources = rag_system.query("What is machine learning?")
        
        # Verify tool was executed
        rag_system.vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )
        
        # Verify sources were captured
        assert len(sources) > 0
        assert "AI Course - Lesson 1|https://example.com/lesson1" in sources
        
        assert response == "Based on search results: ML is..."
    
    def test_query_with_tool_usage_broken_config(self, broken_rag_system):
        """Test query that triggers tool usage with broken config (MAX_RESULTS=0)"""
        rag_system, mock_client = broken_rag_system
        
        # Verify the config is indeed broken
        assert rag_system.vector_store.max_results == 0
        
        # Mock tool use response
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool1"
        tool_block.input = {"query": "machine learning"}
        
        initial_response = Mock()
        initial_response.content = [tool_block]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="I couldn't find relevant information")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock vector store to simulate broken behavior
        from vector_store import SearchResults
        rag_system.vector_store.search = Mock(return_value=SearchResults(
            documents=[],  # Empty due to MAX_RESULTS=0
            metadata=[],
            distances=[]
        ))
        
        response, sources = rag_system.query("What is machine learning?")
        
        # Verify tool was attempted but returned no results
        rag_system.vector_store.search.assert_called_once()
        
        # Verify no sources due to empty results
        assert len(sources) == 0
        
        # The response should indicate failure to find information
        assert response == "I couldn't find relevant information"
    
    def test_get_course_analytics(self, temp_rag_system):
        """Test getting course analytics"""
        rag_system, _ = temp_rag_system
        
        # Mock vector store methods
        rag_system.vector_store.get_course_count = Mock(return_value=3)
        rag_system.vector_store.get_existing_course_titles = Mock(return_value=[
            "Course 1", "Course 2", "Course 3"
        ])
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 3
        assert analytics["course_titles"] == ["Course 1", "Course 2", "Course 3"]
    
    def test_tool_manager_integration(self, temp_rag_system):
        """Test that tool manager properly integrates with search tools"""
        rag_system, _ = temp_rag_system
        
        # Test tool registration
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_definitions) == 2
        
        tool_names = [td["name"] for td in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        
        # Test tool execution
        from vector_store import SearchResults
        rag_system.vector_store.search = Mock(return_value=SearchResults(
            documents=["Test document"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        ))
        
        result = rag_system.tool_manager.execute_tool("search_course_content", query="test")
        
        assert "Test document" in result
        assert "[Test Course - Lesson 1]" in result
    
    def test_source_tracking_and_reset(self, temp_rag_system):
        """Test that sources are properly tracked and reset between queries"""
        rag_system, mock_client = temp_rag_system
        
        # Mock tool use response
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool1"
        tool_block.input = {"query": "test"}
        
        initial_response = Mock()
        initial_response.content = [tool_block]
        initial_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock(text="Response")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock search results
        from vector_store import SearchResults
        rag_system.vector_store.search = Mock(return_value=SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1]
        ))
        rag_system.vector_store.get_lesson_link = Mock(return_value="https://example.com/lesson")
        
        # First query - should have sources
        response1, sources1 = rag_system.query("Query 1")
        assert len(sources1) > 0
        
        # Reset mock for second query
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Second query - should have fresh sources (not cumulative)
        response2, sources2 = rag_system.query("Query 2")
        assert len(sources2) > 0
        
        # Sources should be the same for identical search results
        assert sources1 == sources2
    
    def test_error_handling_in_query(self, temp_rag_system):
        """Test error handling during query processing"""
        rag_system, mock_client = temp_rag_system
        
        # Mock AI to raise exception
        mock_client.messages.create.side_effect = Exception("API Error")
        
        # This should not crash the system but handle gracefully
        with pytest.raises(Exception):
            rag_system.query("What is machine learning?")