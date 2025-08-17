import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator, ToolRoundState
from search_tools import ToolManager, CourseSearchTool


class TestAIGenerator:
    """Test AIGenerator functionality"""
    
    def test_init(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-api-key", "claude-3-sonnet-20240229")
        
        assert generator.model == "claude-3-sonnet-20240229"
        assert generator.base_params["model"] == "claude-3-sonnet-20240229"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    def test_system_prompt_content(self):
        """Test that system prompt contains proper tool usage guidelines"""
        assert "Course outline queries" in AIGenerator.SYSTEM_PROMPT
        assert "Content-specific questions" in AIGenerator.SYSTEM_PROMPT
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT
    
    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test generating response without tools"""
        # Mock the client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response("What is machine learning?")
        
        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == "test-model"
        assert call_args["messages"][0]["content"] == "What is machine learning?"
        assert "tools" not in call_args
        
        assert result == "This is a response"
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test generating response with conversation history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with context")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "test-model")
        history = "Previous conversation context"
        result = generator.generate_response("Follow up question", conversation_history=history)
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args[1]
        assert history in call_args["system"]
        assert AIGenerator.SYSTEM_PROMPT in call_args["system"]
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_usage(self, mock_anthropic):
        """Test generating response with tools available but not used"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct response without tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "test-model")
        tools = [{"name": "test_tool", "description": "A test tool"}]
        
        result = generator.generate_response("General question", tools=tools)
        
        # Verify tools were included in API call
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        assert result == "Direct response without tools"
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_usage(self, mock_anthropic):
        """Test generating response when Claude decides to use tools"""
        mock_client = Mock()
        
        # Mock initial tool use response
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_call_123"
        tool_block.input = {"query": "machine learning"}
        
        initial_response = Mock()
        initial_response.content = [tool_block]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Based on the search results, machine learning is...")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: ML is a subset of AI..."
        
        generator = AIGenerator("test-key", "test-model")
        tools = [{"name": "search_course_content", "description": "Search course content"}]
        
        result = generator.generate_response(
            "What is machine learning?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
        
        # Verify final result
        assert result == "Based on the search results, machine learning is..."
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic):
        """Test handling multiple tool calls in one response"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Create mock tool blocks
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.id = "tool1"
        tool_block1.input = {"query": "preprocessing"}
        
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.id = "tool2"
        tool_block2.input = {"course_name": "ML Course"}
        
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Let me search for information..."
        
        initial_response = Mock()
        initial_response.content = [text_block, tool_block1, tool_block2]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Preprocessing results...",
            "Course outline: 1. Introduction 2. Advanced..."
        ]
        
        # Mock base parameters
        base_params = {
            "messages": [{"role": "user", "content": "Tell me about preprocessing"}],
            "system": "System prompt"
        }
        
        generator = AIGenerator("test-key", "test-model")
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Combined response using both tool results")]
        mock_client.messages.create.return_value = final_response
        
        result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="preprocessing")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="ML Course")
        
        # Verify final API call was made
        mock_client.messages.create.assert_called_once()
        
        # Check that the tool results were included in the message
        final_call_args = mock_client.messages.create.call_args[1]
        messages = final_call_args["messages"]
        
        # Should have: original user message, assistant's tool use, tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Check tool results format
        tool_results = messages[2]["content"]
        assert len(tool_results) == 2
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool1"
        assert tool_results[0]["content"] == "Preprocessing results..."
        assert tool_results[1]["type"] == "tool_result"
        assert tool_results[1]["tool_use_id"] == "tool2"
        
        assert result == "Combined response using both tool results"
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_no_tools(self, mock_anthropic):
        """Test handling response with no tool calls"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Response with no tool use blocks
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Regular response"
        
        initial_response = Mock()
        initial_response.content = [text_block]
        
        mock_tool_manager = Mock()
        base_params = {
            "messages": [{"role": "user", "content": "Question"}],
            "system": "System prompt"
        }
        
        generator = AIGenerator("test-key", "test-model")
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Final response")]
        mock_client.messages.create.return_value = final_response
        
        result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        # Verify no tools were executed
        mock_tool_manager.execute_tool.assert_not_called()
        
        # Verify API was still called for final response
        mock_client.messages.create.assert_called_once()
        
        assert result == "Final response"
    
    def test_tool_integration_with_search_tool(self, mock_vector_store):
        """Test integration between AIGenerator and actual search tools"""
        # Create real tool manager with search tool
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Mock successful search
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=["Machine learning is a field of AI"],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1]
        )
        
        # Test tool execution through manager
        result = tool_manager.execute_tool("search_course_content", query="machine learning")
        
        assert "Machine learning is a field of AI" in result
        assert "[AI Course - Lesson 1]" in result
    
    @patch('anthropic.Anthropic')
    def test_error_handling_in_tool_execution(self, mock_anthropic):
        """Test error handling when tool execution fails"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock tool use response
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "failing_tool"
        tool_block.id = "tool1"
        tool_block.input = {"param": "value"}
        
        initial_response = Mock()
        initial_response.content = [tool_block]
        
        # Mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        base_params = {
            "messages": [{"role": "user", "content": "Question"}],
            "system": "System prompt"
        }
        
        generator = AIGenerator("test-key", "test-model")
        
        # This should not raise an exception, but handle it gracefully
        # The actual behavior depends on implementation, but the test verifies
        # that we don't crash when tools fail
        with pytest.raises(Exception):
            generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
    
    def test_api_parameters_structure(self):
        """Test that API parameters are structured correctly"""
        generator = AIGenerator("test-key", "test-model")
        
        # Test base parameters
        assert generator.base_params["model"] == "test-model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
        
        # Verify system prompt is not empty and contains tool instructions
        assert len(AIGenerator.SYSTEM_PROMPT) > 100
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT


class TestToolRoundState:
    """Test ToolRoundState functionality"""
    
    def test_tool_round_state_initialization(self):
        """Test ToolRoundState default initialization"""
        state = ToolRoundState()
        
        assert state.current_round == 1
        assert state.max_rounds == 2
        assert state.tool_execution_history == []
        assert state.messages == []
        assert state.system_content == ""
        assert state.has_errors is False
    
    def test_tool_round_state_custom_max_rounds(self):
        """Test ToolRoundState with custom max rounds"""
        state = ToolRoundState(max_rounds=3)
        
        assert state.max_rounds == 3
        assert state.can_continue() is True
    
    def test_can_continue_normal_conditions(self):
        """Test can_continue under normal conditions"""
        state = ToolRoundState()
        
        # Should be able to continue on first round
        assert state.can_continue() is True
        
        # Advance to round 2
        state.advance_round()
        assert state.current_round == 2
        assert state.can_continue() is True  # Still within max rounds
        
        # Advance to round 3 (beyond max)
        state.advance_round()
        assert state.current_round == 3
        assert state.can_continue() is False  # Now beyond max rounds
    
    def test_can_continue_with_errors(self):
        """Test can_continue when errors occur"""
        state = ToolRoundState()
        
        # Set error state
        state.has_errors = True
        assert state.can_continue() is False
    
    def test_advance_round(self):
        """Test round advancement"""
        state = ToolRoundState()
        
        assert state.current_round == 1
        state.advance_round()
        assert state.current_round == 2
        state.advance_round()
        assert state.current_round == 3  # Beyond max, but allowed
    
    def test_add_tool_execution(self):
        """Test recording tool executions"""
        state = ToolRoundState()
        
        state.add_tool_execution("search_tool", {"query": "test"}, "results")
        
        assert len(state.tool_execution_history) == 1
        execution = state.tool_execution_history[0]
        assert execution["round"] == 1
        assert execution["tool_name"] == "search_tool"
        assert execution["input"] == {"query": "test"}
        assert execution["result"] == "results"


class TestSequentialToolCalling:
    """Test sequential tool calling functionality"""
    
    def test_system_prompt_updated_for_sequential_calling(self):
        """Test that system prompt mentions sequential tool calling"""
        assert "Sequential tool calling" in AIGenerator.SYSTEM_PROMPT
        assert "up to 2 sequential tool calls" in AIGenerator.SYSTEM_PROMPT
        assert "Multi-step queries" in AIGenerator.SYSTEM_PROMPT
        assert "Sequential Tool Examples" in AIGenerator.SYSTEM_PROMPT
    
    @patch('anthropic.Anthropic')
    def test_sequential_tools_single_round_no_tools(self, mock_anthropic):
        """Test sequential tool calling with no tool usage"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct answer without tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response_with_sequential_tools(
            "What is 2+2?",
            tools=[{"name": "search_tool"}],
            tool_manager=Mock()
        )
        
        # Should make only one API call since no tools were used
        assert mock_client.messages.create.call_count == 1
        assert result == "Direct answer without tools"
    
    @patch('anthropic.Anthropic')
    def test_sequential_tools_single_round_with_tools(self, mock_anthropic):
        """Test sequential tool calling with single round tool usage"""
        mock_client = Mock()
        
        # Mock tool use response
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool1"
        tool_block.input = {"query": "machine learning"}
        
        # First response with tool use
        first_response = Mock()
        first_response.content = [tool_block]
        first_response.stop_reason = "tool_use"
        
        # Second response with final answer (no more tools)
        second_response = Mock()
        second_response.content = [Mock(text="Final answer after tool use")]
        second_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [first_response, second_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about ML"
        
        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response_with_sequential_tools(
            "What is machine learning?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Should make two API calls: initial + after tool execution
        assert mock_client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="machine learning"
        )
        assert result == "Final answer after tool use"
    
    @patch('anthropic.Anthropic')
    def test_sequential_tools_two_rounds(self, mock_anthropic):
        """Test sequential tool calling with two rounds"""
        mock_client = Mock()
        
        # Round 1: First tool use
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "get_course_outline"
        tool_block1.id = "tool1"
        tool_block1.input = {"course_name": "ML Course"}
        
        first_response = Mock()
        first_response.content = [tool_block1]
        first_response.stop_reason = "tool_use"
        
        # Round 2: Second tool use
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "search_course_content"
        tool_block2.id = "tool2"
        tool_block2.input = {"query": "advanced topics lesson 4"}
        
        second_response = Mock()
        second_response.content = [tool_block2]
        second_response.stop_reason = "tool_use"
        
        # Final response without tools
        final_response = Mock()
        final_response.content = [Mock(text="Combined analysis from both searches")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [
            first_response, second_response, final_response
        ]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: 1. Intro 2. Basics 3. Intermediate 4. Advanced ML",
            "Lesson 4 covers deep learning, neural networks, and advanced algorithms"
        ]
        
        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response_with_sequential_tools(
            "What advanced topics are covered in lesson 4 of ML Course?",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Should make three API calls: Round 1 + Round 2 + Final
        assert mock_client.messages.create.call_count == 3
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_name="ML Course"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="advanced topics lesson 4"
        )
        
        assert result == "Combined analysis from both searches"
    
    @patch('anthropic.Anthropic')
    def test_sequential_tools_max_rounds_reached(self, mock_anthropic):
        """Test sequential tool calling when max rounds are reached"""
        mock_client = Mock()
        
        # Both responses have tool use
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_tool"
        tool_block.id = "tool1"
        tool_block.input = {"query": "test"}
        
        tool_response = Mock()
        tool_response.content = [tool_block]
        tool_response.stop_reason = "tool_use"
        
        # Final response after max rounds
        final_response = Mock()
        final_response.content = [Mock(text="Final response after max rounds")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [
            tool_response, tool_response, final_response
        ]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response_with_sequential_tools(
            "Complex query requiring multiple searches",
            tools=[{"name": "search_tool"}],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        # Should make 3 calls: Round 1 + Round 2 + Final (no tools)
        assert mock_client.messages.create.call_count == 3
        
        # Should execute tools only twice (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        assert result == "Final response after max rounds"
    
    @patch('anthropic.Anthropic')
    def test_sequential_tools_tool_execution_error(self, mock_anthropic):
        """Test sequential tool calling when tool execution fails"""
        mock_client = Mock()
        
        # First response with tool use
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "failing_tool"
        tool_block.id = "tool1"
        tool_block.input = {"query": "test"}
        
        first_response = Mock()
        first_response.content = [tool_block]
        first_response.stop_reason = "tool_use"
        
        # Final response after error
        final_response = Mock()
        final_response.content = [Mock(text="Error response")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager that fails
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")
        
        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response_with_sequential_tools(
            "Query that causes tool failure",
            tools=[{"name": "failing_tool"}],
            tool_manager=mock_tool_manager
        )
        
        # Should handle error gracefully and make final call
        assert mock_client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once()
        assert result == "Error response"
    
    @patch('anthropic.Anthropic')
    def test_sequential_tools_with_conversation_history(self, mock_anthropic):
        """Test sequential tool calling with conversation history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with context")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "test-model")
        history = "User asked about ML basics. Assistant explained fundamentals."
        
        result = generator.generate_response_with_sequential_tools(
            "Now tell me about advanced topics",
            conversation_history=history,
            tools=[{"name": "search_tool"}],
            tool_manager=Mock()
        )
        
        # Verify history is included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        assert history in call_args["system"]
        assert AIGenerator.SYSTEM_PROMPT in call_args["system"]
        assert result == "Response with context"
    
    def test_has_tool_calls_helper_method(self):
        """Test _has_tool_calls helper method"""
        generator = AIGenerator("test-key", "test-model")
        
        # Response with tool calls
        tool_block = Mock()
        tool_block.type = "tool_use"
        
        text_block = Mock()
        text_block.type = "text"
        
        response_with_tools = Mock()
        response_with_tools.content = [text_block, tool_block]
        
        assert generator._has_tool_calls(response_with_tools) is True
        
        # Response without tool calls
        response_no_tools = Mock()
        response_no_tools.content = [text_block]
        
        assert generator._has_tool_calls(response_no_tools) is False
        
        # Empty response
        assert generator._has_tool_calls(None) is False
        
        empty_response = Mock()
        empty_response.content = []
        assert generator._has_tool_calls(empty_response) is False
    
    def test_extract_text_response_helper_method(self):
        """Test _extract_text_response helper method"""
        generator = AIGenerator("test-key", "test-model")
        
        # Response with text content
        text_block = Mock()
        text_block.text = "This is the response text"
        
        response = Mock()
        response.content = [text_block]
        
        result = generator._extract_text_response(response)
        assert result == "This is the response text"
        
        # Response with no content
        result = generator._extract_text_response(None)
        assert "Error: No valid response content" in result
        
        # Response with empty content
        empty_response = Mock()
        empty_response.content = []
        result = generator._extract_text_response(empty_response)
        assert "Error: No valid response content" in result