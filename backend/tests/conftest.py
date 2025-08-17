import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config

@pytest.fixture
def sample_course():
    """Create a sample course with lessons for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Jane Smith",
        lessons=[
            Lesson(lesson_number=1, title="What is Machine Learning?", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Types of Learning", lesson_link="https://example.com/lesson2"),
            Lesson(lesson_number=3, title="Data Preprocessing", lesson_link="https://example.com/lesson3")
        ]
    )

@pytest.fixture
def sample_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="There are three main types of machine learning: supervised, unsupervised, and reinforcement learning.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Data preprocessing is crucial for machine learning success. This includes cleaning, transforming, and preparing data.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        )
    ]

@pytest.fixture
def mock_search_results():
    """Create mock search results for testing"""
    return SearchResults(
        documents=[
            "Machine learning is a subset of artificial intelligence.",
            "Data preprocessing is crucial for ML success."
        ],
        metadata=[
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Introduction to Machine Learning", "lesson_number": 3, "chunk_index": 2}
        ],
        distances=[0.2, 0.4]
    )

@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )

@pytest.fixture
def error_search_results():
    """Create error search results for testing"""
    return SearchResults.empty("Search failed: No results found")

@pytest.fixture
def temp_chroma_path():
    """Create temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_config():
    """Create test configuration with safe defaults"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-key-not-real"
    config.CHROMA_PATH = tempfile.mkdtemp()
    config.MAX_RESULTS = 5  # Set to working value for most tests
    config.CHUNK_SIZE = 100
    config.CHUNK_OVERLAP = 20
    config.MAX_HISTORY = 2
    return config

@pytest.fixture
def broken_config():
    """Create config with the current broken MAX_RESULTS setting"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-key-not-real"
    config.CHROMA_PATH = tempfile.mkdtemp()
    config.MAX_RESULTS = 0  # This is the current broken setting
    config.CHUNK_SIZE = 100
    config.CHUNK_OVERLAP = 20
    config.MAX_HISTORY = 2
    return config

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing without API calls"""
    client = Mock()
    
    # Mock a standard text response
    text_response = Mock()
    text_response.content = [Mock(text="This is a test response")]
    text_response.stop_reason = "end_turn"
    
    # Mock a tool use response
    tool_response = Mock()
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "test-tool-id"
    tool_block.input = {"query": "test query"}
    tool_response.content = [tool_block]
    tool_response.stop_reason = "tool_use"
    
    client.messages.create.side_effect = [tool_response, text_response]
    return client

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    store = Mock(spec=VectorStore)
    store.max_results = 5
    
    # Default successful search behavior
    store.search.return_value = SearchResults(
        documents=["Sample document content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    )
    
    # Mock other methods
    store.get_course_link.return_value = "https://example.com/course"
    store.get_lesson_link.return_value = "https://example.com/lesson1"
    store._resolve_course_name.return_value = "Test Course"
    
    # Mock course_catalog collection
    store.course_catalog = Mock()
    store.course_catalog.get.return_value = {
        'metadatas': [{
            'title': 'Test Course',
            'course_link': 'https://example.com/course',
            'instructor': 'John Doe',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction"}]'
        }]
    }
    
    return store

class MockEmbeddingFunction:
    """Mock embedding function for ChromaDB that implements required interface"""
    
    def __call__(self, input):
        """Make the function callable"""
        if isinstance(input, str):
            input = [input]
        return [[0.1] * 384 for _ in input]
    
    def name(self):
        """Return a name for the embedding function"""
        return "mock-embedding-function"
    
    def is_legacy(self):
        """Return False to indicate this is not a legacy function"""
        return False

class MockSentenceTransformer:
    """Mock sentence transformer for testing without downloading models"""
    def encode(self, texts):
        # Return dummy embeddings
        if isinstance(texts, str):
            return [0.1] * 384
        return [[0.1] * 384 for _ in texts]

@pytest.fixture
def mock_embedding_function():
    """Mock embedding function for ChromaDB testing"""
    return MockEmbeddingFunction()