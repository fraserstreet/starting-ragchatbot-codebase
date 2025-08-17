"""Tests to catch specific bugs in AI generator that cause runtime errors"""
import pytest
from unittest.mock import Mock, patch
from ai_generator import AIGenerator


class TestAIGeneratorBugs:
    """Test AI generator bugs that cause runtime errors"""
    
    @patch('anthropic.Anthropic')
    def test_empty_content_array_bug(self, mock_anthropic):
        """Test handling of empty content array from Anthropic API"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response with empty content array (this can happen in error cases)
        mock_response = Mock()
        mock_response.content = []  # Empty content array - this causes IndexError
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "test-model")
        
        # This should now handle gracefully instead of crashing
        result = generator.generate_response("What is machine learning?")
        assert "Error: No valid response content received" in result
    
    @patch('anthropic.Anthropic')
    def test_empty_content_array_in_tool_execution(self, mock_anthropic):
        """Test handling of empty content array in final response after tool execution"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock initial tool use response
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool1"
        tool_block.input = {"query": "test"}
        
        initial_response = Mock()
        initial_response.content = [tool_block]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response with empty content array
        final_response = Mock()
        final_response.content = []  # Empty content array - this causes IndexError
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        generator = AIGenerator("test-key", "test-model")
        tools = [{"name": "search_course_content", "description": "Search"}]
        
        # This should now handle gracefully instead of crashing
        result = generator.generate_response(
            "Search for something", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        assert "Error: No valid response content received" in result
    
    @patch('anthropic.Anthropic')
    def test_non_text_content_block(self, mock_anthropic):
        """Test handling of non-text content blocks"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response with non-text content block
        mock_content_block = Mock()
        mock_content_block.type = "image"  # Not a text block
        # This might not have a .text attribute
        
        mock_response = Mock()
        mock_response.content = [mock_content_block]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "test-model")
        
        # This might cause AttributeError if we try to access .text on a non-text block
        with pytest.raises(AttributeError):
            generator.generate_response("What is machine learning?")
    
    @patch('anthropic.Anthropic')
    def test_none_content_block(self, mock_anthropic):
        """Test handling when content block is None"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response with None content block
        mock_response = Mock()
        mock_response.content = [None]  # None content block
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "test-model")
        
        # This might cause AttributeError if we try to access .text on None
        with pytest.raises(AttributeError):
            generator.generate_response("What is machine learning?")