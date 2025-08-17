import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_successful_search(self, mock_vector_store, mock_search_results):
        """Test successful search execution with results"""
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("machine learning basics")
        
        # Verify search was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="machine learning basics",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result formatting
        assert "[Introduction to Machine Learning - Lesson 1]" in result
        assert "Machine learning is a subset of artificial intelligence." in result
        assert len(tool.last_sources) == 2
        assert "Introduction to Machine Learning - Lesson 1|https://example.com/lesson1" in tool.last_sources
    
    def test_execute_with_course_filter(self, mock_vector_store, mock_search_results):
        """Test search execution with course name filter"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("data preprocessing", course_name="ML Course")
        
        mock_vector_store.search.assert_called_once_with(
            query="data preprocessing",
            course_name="ML Course",
            lesson_number=None
        )
    
    def test_execute_with_lesson_filter(self, mock_vector_store, mock_search_results):
        """Test search execution with lesson number filter"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("machine learning", lesson_number=2)
        
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=2
        )
    
    def test_execute_with_both_filters(self, mock_vector_store, mock_search_results):
        """Test search execution with both course and lesson filters"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("preprocessing", course_name="ML Course", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="preprocessing",
            course_name="ML Course",
            lesson_number=3
        )
    
    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("nonexistent topic")
        
        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0
    
    def test_execute_empty_results_with_filters(self, mock_vector_store, empty_search_results):
        """Test handling of empty results with filters"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("topic", course_name="Some Course", lesson_number=5)
        
        assert "No relevant content found in course 'Some Course' in lesson 5" in result
    
    def test_execute_error_handling(self, mock_vector_store, error_search_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = error_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("some query")
        
        assert "Search failed: No results found" in result
    
    def test_format_results_with_links(self, mock_vector_store):
        """Test result formatting with lesson and course links"""
        # Mock search results with different metadata combinations
        results = SearchResults(
            documents=["Content 1", "Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None},
                {"course_title": "Course C", "lesson_number": 2}
            ],
            distances=[0.1, 0.2, 0.3]
        )
        
        # Mock link retrieval
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",  # For Course A, Lesson 1
            None,  # For Course C, Lesson 2 (no link)
        ]
        mock_vector_store.get_course_link.return_value = "https://example.com/courseB"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool._format_results(results)
        
        # Verify formatting
        assert "[Course A - Lesson 1]" in result
        assert "[Course B]" in result  # No lesson number
        assert "[Course C - Lesson 2]" in result
        
        # Verify sources with links
        sources = tool.last_sources
        assert "Course A - Lesson 1|https://example.com/lesson1" in sources
        assert "Course B|https://example.com/courseB" in sources
        assert "Course C - Lesson 2" in sources  # No link for this one
    
    def test_source_tracking_reset(self, mock_vector_store, mock_search_results):
        """Test that sources are properly tracked and can be reset"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # First search
        tool.execute("query 1")
        assert len(tool.last_sources) > 0
        
        # Reset sources
        tool.last_sources = []
        assert len(tool.last_sources) == 0
        
        # Second search
        tool.execute("query 2")
        assert len(tool.last_sources) > 0


class TestCourseOutlineTool:
    """Test CourseOutlineTool functionality"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]
    
    def test_execute_successful_outline(self, mock_vector_store):
        """Test successful course outline retrieval"""
        # Mock course resolution
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        
        # Mock course catalog results
        mock_results = {
            'metadatas': [{
                'title': 'Test Course',
                'course_link': 'https://example.com/course',
                'instructor': 'John Doe',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction"}, {"lesson_number": 2, "lesson_title": "Advanced Topics"}]'
            }]
        }
        mock_vector_store.course_catalog.get.return_value = mock_results
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Test Course")
        
        # Verify course resolution was called
        mock_vector_store._resolve_course_name.assert_called_once_with("Test Course")
        
        # Verify result content
        assert "**Course Title:** Test Course" in result
        assert "**Course Link:** https://example.com/course" in result
        assert "**Course Instructor:** John Doe" in result
        assert "**Number of Lessons:** 2" in result
        assert "1. Introduction" in result
        assert "2. Advanced Topics" in result
    
    def test_execute_course_not_found(self, mock_vector_store):
        """Test handling when course is not found"""
        mock_vector_store._resolve_course_name.return_value = None
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Nonexistent Course")
        
        assert "No course found matching 'Nonexistent Course'" in result
    
    def test_execute_metadata_not_found(self, mock_vector_store):
        """Test handling when course metadata is not found"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.course_catalog.get.return_value = {'metadatas': []}
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Test Course")
        
        assert "Course metadata not found for 'Test Course'" in result
    
    def test_execute_invalid_json(self, mock_vector_store):
        """Test handling of invalid JSON in lessons data"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        
        mock_results = {
            'metadatas': [{
                'title': 'Test Course',
                'lessons_json': 'invalid json'
            }]
        }
        mock_vector_store.course_catalog.get.return_value = mock_results
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Test Course")
        
        assert "Error parsing lesson data" in result


class TestToolManager:
    """Test ToolManager functionality"""
    
    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool
    
    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 2
        tool_names = [d["name"] for d in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
    
    def test_execute_tool(self, mock_vector_store, mock_search_results):
        """Test tool execution via manager"""
        mock_vector_store.search.return_value = mock_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert "Machine learning is a subset of artificial intelligence." in result
    
    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test execution of nonexistent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self, mock_vector_store, mock_search_results):
        """Test getting sources from last search"""
        mock_vector_store.search.return_value = mock_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="test query")
        
        sources = manager.get_last_sources()
        assert len(sources) > 0
    
    def test_reset_sources(self, mock_vector_store, mock_search_results):
        """Test resetting sources"""
        mock_vector_store.search.return_value = mock_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="test query")
        assert len(manager.get_last_sources()) > 0
        
        # Reset sources
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0
    
    def test_register_tool_without_name(self, mock_vector_store):
        """Test registering tool without name raises error"""
        manager = ToolManager()
        
        # Create a mock tool that returns definition without name
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "test"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name' in its definition"):
            manager.register_tool(mock_tool)