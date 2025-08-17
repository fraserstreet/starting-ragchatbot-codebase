import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test SearchResults utility class"""
    
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course': 'A'}, {'course': 'B'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'course': 'A'}, {'course': 'B'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.is_empty()
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Test error")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error"
        assert results.is_empty()
    
    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(['doc'], [{}], [0.1])
        
        assert empty_results.is_empty()
        assert not non_empty_results.is_empty()


class TestVectorStore:
    """Test VectorStore functionality"""
    
    @pytest.fixture
    def temp_vector_store(self, mock_embedding_function):
        """Create a temporary VectorStore for testing"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Mock the embedding function to avoid downloading models
            with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_ef:
                mock_ef.return_value = mock_embedding_function
                
                store = VectorStore(temp_dir, "test-model", max_results=5)
                yield store
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_init_creates_collections(self, temp_vector_store):
        """Test that VectorStore initializes with proper collections"""
        store = temp_vector_store
        
        assert store.course_catalog is not None
        assert store.course_content is not None
        assert store.max_results == 5
    
    def test_search_with_max_results_zero(self, test_config, broken_config, mock_embedding_function):
        """Test search behavior when MAX_RESULTS is 0 (current broken config)"""
        temp_dir = tempfile.mkdtemp()
        try:
            with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_ef:
                mock_ef.return_value = mock_embedding_function
                
                # Create store with broken config (MAX_RESULTS = 0)
                store = VectorStore(temp_dir, "test-model", max_results=broken_config.MAX_RESULTS)
                
                # Mock the collection query to simulate ChromaDB behavior with n_results=0
                store.course_content.query = Mock(return_value={
                    'documents': [[]],  # Empty results when n_results=0
                    'metadatas': [[]],
                    'distances': [[]]
                })
                
                results = store.search("test query")
                
                # Verify that search was called with n_results=0
                store.course_content.query.assert_called_once_with(
                    query_texts=["test query"],
                    n_results=0,  # This is the problem!
                    where=None
                )
                
                # Verify results are empty
                assert results.is_empty()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_search_with_proper_max_results(self, test_config, mock_embedding_function):
        """Test search behavior when MAX_RESULTS is properly set"""
        temp_dir = tempfile.mkdtemp()
        try:
            with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_ef:
                mock_ef.return_value = mock_embedding_function
                
                # Create store with working config (MAX_RESULTS = 5)
                store = VectorStore(temp_dir, "test-model", max_results=test_config.MAX_RESULTS)
                
                # Mock successful search results
                store.course_content.query = Mock(return_value={
                    'documents': [['Sample document content']],
                    'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
                    'distances': [[0.2]]
                })
                
                results = store.search("test query")
                
                # Verify that search was called with n_results=5
                store.course_content.query.assert_called_once_with(
                    query_texts=["test query"],
                    n_results=5,
                    where=None
                )
                
                # Verify results are not empty
                assert not results.is_empty()
                assert len(results.documents) == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_search_with_course_filter(self, temp_vector_store):
        """Test search with course name filter"""
        store = temp_vector_store
        
        # Mock course name resolution
        store._resolve_course_name = Mock(return_value="Resolved Course")
        
        # Mock search results
        store.course_content.query = Mock(return_value={
            'documents': [['Content']],
            'metadatas': [[{'course_title': 'Resolved Course'}]],
            'distances': [[0.1]]
        })
        
        results = store.search("query", course_name="Partial Course Name")
        
        # Verify course name was resolved
        store._resolve_course_name.assert_called_once_with("Partial Course Name")
        
        # Verify search was called with proper filter
        store.course_content.query.assert_called_once_with(
            query_texts=["query"],
            n_results=5,
            where={"course_title": "Resolved Course"}
        )
    
    def test_search_with_lesson_filter(self, temp_vector_store):
        """Test search with lesson number filter"""
        store = temp_vector_store
        
        store.course_content.query = Mock(return_value={
            'documents': [['Content']],
            'metadatas': [[{'lesson_number': 2}]],
            'distances': [[0.1]]
        })
        
        results = store.search("query", lesson_number=2)
        
        store.course_content.query.assert_called_once_with(
            query_texts=["query"],
            n_results=5,
            where={"lesson_number": 2}
        )
    
    def test_search_with_both_filters(self, temp_vector_store):
        """Test search with both course and lesson filters"""
        store = temp_vector_store
        
        store._resolve_course_name = Mock(return_value="Test Course")
        store.course_content.query = Mock(return_value={
            'documents': [['Content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 3}]],
            'distances': [[0.1]]
        })
        
        results = store.search("query", course_name="Test Course", lesson_number=3)
        
        expected_filter = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 3}
            ]
        }
        
        store.course_content.query.assert_called_once_with(
            query_texts=["query"],
            n_results=5,
            where=expected_filter
        )
    
    def test_search_course_not_found(self, temp_vector_store):
        """Test search when course name cannot be resolved"""
        store = temp_vector_store
        store._resolve_course_name = Mock(return_value=None)
        
        results = store.search("query", course_name="Nonexistent Course")
        
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()
    
    def test_search_exception_handling(self, temp_vector_store):
        """Test search exception handling"""
        store = temp_vector_store
        store.course_content.query = Mock(side_effect=Exception("ChromaDB error"))
        
        results = store.search("query")
        
        assert "Search error: ChromaDB error" in results.error
        assert results.is_empty()
    
    def test_resolve_course_name_success(self, temp_vector_store):
        """Test successful course name resolution"""
        store = temp_vector_store
        
        # Mock course catalog query
        store.course_catalog.query = Mock(return_value={
            'documents': [['Course Title']],
            'metadatas': [[{'title': 'Full Course Title'}]]
        })
        
        result = store._resolve_course_name("Partial Name")
        
        store.course_catalog.query.assert_called_once_with(
            query_texts=["Partial Name"],
            n_results=1
        )
        assert result == "Full Course Title"
    
    def test_resolve_course_name_not_found(self, temp_vector_store):
        """Test course name resolution when no match found"""
        store = temp_vector_store
        
        store.course_catalog.query = Mock(return_value={
            'documents': [[]],
            'metadatas': [[]]
        })
        
        result = store._resolve_course_name("Nonexistent")
        assert result is None
    
    def test_resolve_course_name_exception(self, temp_vector_store):
        """Test course name resolution exception handling"""
        store = temp_vector_store
        store.course_catalog.query = Mock(side_effect=Exception("Query failed"))
        
        result = store._resolve_course_name("Any Name")
        assert result is None
    
    def test_build_filter_no_filters(self, temp_vector_store):
        """Test filter building with no filters"""
        store = temp_vector_store
        result = store._build_filter(None, None)
        assert result is None
    
    def test_build_filter_course_only(self, temp_vector_store):
        """Test filter building with course filter only"""
        store = temp_vector_store
        result = store._build_filter("Test Course", None)
        assert result == {"course_title": "Test Course"}
    
    def test_build_filter_lesson_only(self, temp_vector_store):
        """Test filter building with lesson filter only"""
        store = temp_vector_store
        result = store._build_filter(None, 2)
        assert result == {"lesson_number": 2}
    
    def test_build_filter_both(self, temp_vector_store):
        """Test filter building with both filters"""
        store = temp_vector_store
        result = store._build_filter("Test Course", 3)
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 3}
            ]
        }
        assert result == expected
    
    def test_add_course_metadata(self, temp_vector_store, sample_course):
        """Test adding course metadata"""
        store = temp_vector_store
        store.course_catalog.add = Mock()
        
        store.add_course_metadata(sample_course)
        
        # Verify add was called with correct parameters
        args, kwargs = store.course_catalog.add.call_args
        assert kwargs['documents'] == [sample_course.title]
        assert kwargs['ids'] == [sample_course.title]
        
        metadata = kwargs['metadatas'][0]
        assert metadata['title'] == sample_course.title
        assert metadata['instructor'] == sample_course.instructor
        assert metadata['course_link'] == sample_course.course_link
        assert metadata['lesson_count'] == len(sample_course.lessons)
        assert 'lessons_json' in metadata
    
    def test_add_course_content(self, temp_vector_store, sample_chunks):
        """Test adding course content chunks"""
        store = temp_vector_store
        store.course_content.add = Mock()
        
        store.add_course_content(sample_chunks)
        
        # Verify add was called with correct parameters
        args, kwargs = store.course_content.add.call_args
        
        assert len(kwargs['documents']) == len(sample_chunks)
        assert len(kwargs['metadatas']) == len(sample_chunks)
        assert len(kwargs['ids']) == len(sample_chunks)
        
        # Check first chunk
        assert kwargs['documents'][0] == sample_chunks[0].content
        assert kwargs['metadatas'][0]['course_title'] == sample_chunks[0].course_title
        assert kwargs['metadatas'][0]['lesson_number'] == sample_chunks[0].lesson_number
    
    def test_add_course_content_empty(self, temp_vector_store):
        """Test adding empty course content list"""
        store = temp_vector_store
        store.course_content.add = Mock()
        
        store.add_course_content([])
        
        # Verify add was not called
        store.course_content.add.assert_not_called()
    
    def test_clear_all_data(self, temp_vector_store):
        """Test clearing all data"""
        store = temp_vector_store
        
        # Mock the client methods
        store.client.delete_collection = Mock()
        store._create_collection = Mock()
        
        store.clear_all_data()
        
        # Verify collections were deleted
        assert store.client.delete_collection.call_count == 2
        store.client.delete_collection.assert_any_call("course_catalog")
        store.client.delete_collection.assert_any_call("course_content")
        
        # Verify collections were recreated
        assert store._create_collection.call_count == 2
    
    def test_get_existing_course_titles(self, temp_vector_store):
        """Test getting existing course titles"""
        store = temp_vector_store
        store.course_catalog.get = Mock(return_value={
            'ids': ['Course 1', 'Course 2', 'Course 3']
        })
        
        titles = store.get_existing_course_titles()
        
        assert titles == ['Course 1', 'Course 2', 'Course 3']
        store.course_catalog.get.assert_called_once()
    
    def test_get_existing_course_titles_empty(self, temp_vector_store):
        """Test getting course titles when none exist"""
        store = temp_vector_store
        store.course_catalog.get = Mock(return_value=None)
        
        titles = store.get_existing_course_titles()
        
        assert titles == []
    
    def test_get_existing_course_titles_exception(self, temp_vector_store):
        """Test getting course titles with exception"""
        store = temp_vector_store
        store.course_catalog.get = Mock(side_effect=Exception("Database error"))
        
        titles = store.get_existing_course_titles()
        
        assert titles == []
    
    def test_get_course_count(self, temp_vector_store):
        """Test getting course count"""
        store = temp_vector_store
        store.course_catalog.get = Mock(return_value={
            'ids': ['Course 1', 'Course 2']
        })
        
        count = store.get_course_count()
        
        assert count == 2
    
    def test_get_course_count_empty(self, temp_vector_store):
        """Test getting course count when no courses exist"""
        store = temp_vector_store
        store.course_catalog.get = Mock(return_value=None)
        
        count = store.get_course_count()
        
        assert count == 0
    
    def test_get_course_link(self, temp_vector_store):
        """Test getting course link"""
        store = temp_vector_store
        store.course_catalog.get = Mock(return_value={
            'metadatas': [{'course_link': 'https://example.com/course'}]
        })
        
        link = store.get_course_link("Test Course")
        
        store.course_catalog.get.assert_called_once_with(ids=["Test Course"])
        assert link == 'https://example.com/course'
    
    def test_get_course_link_not_found(self, temp_vector_store):
        """Test getting course link when course not found"""
        store = temp_vector_store
        store.course_catalog.get = Mock(return_value=None)
        
        link = store.get_course_link("Nonexistent Course")
        
        assert link is None
    
    def test_get_lesson_link(self, temp_vector_store):
        """Test getting lesson link"""
        store = temp_vector_store
        lessons_json = '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}, {"lesson_number": 2, "lesson_link": "https://example.com/lesson2"}]'
        store.course_catalog.get = Mock(return_value={
            'metadatas': [{'lessons_json': lessons_json}]
        })
        
        link = store.get_lesson_link("Test Course", 2)
        
        store.course_catalog.get.assert_called_once_with(ids=["Test Course"])
        assert link == 'https://example.com/lesson2'
    
    def test_get_lesson_link_not_found(self, temp_vector_store):
        """Test getting lesson link when lesson not found"""
        store = temp_vector_store
        lessons_json = '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}]'
        store.course_catalog.get = Mock(return_value={
            'metadatas': [{'lessons_json': lessons_json}]
        })
        
        link = store.get_lesson_link("Test Course", 5)  # Lesson 5 doesn't exist
        
        assert link is None