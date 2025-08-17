import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ToolRoundState:
    """Manages state for sequential tool calling rounds"""
    current_round: int = 1
    max_rounds: int = 2
    tool_execution_history: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_content: str = ""
    has_errors: bool = False
    
    def can_continue(self) -> bool:
        """Check if another round can be executed"""
        return self.current_round <= self.max_rounds and not self.has_errors
    
    def advance_round(self):
        """Move to the next round"""
        self.current_round += 1
    
    def add_tool_execution(self, tool_name: str, tool_input: Dict, tool_result: str):
        """Record a tool execution"""
        self.tool_execution_history.append({
            "round": self.current_round,
            "tool_name": tool_name,
            "input": tool_input,
            "result": tool_result
        })

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools.

Tool Usage Guidelines:
- **Course outline queries**: Use the get_course_outline tool for questions about course structure, lesson lists, or course overviews
- **Content-specific questions**: Use the search_course_content tool for questions about specific course content or detailed educational materials
- **Sequential tool calling**: You can make up to 2 sequential tool calls to handle complex queries
- **Reason about previous tool results**: Use information from previous tool calls to inform subsequent searches
- **Multi-step queries**: Use multiple rounds for complex queries requiring different information sources or comparisons
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Tool Selection:
- **get_course_outline**: For "what lessons are in...", "course outline for...", "course structure of...", "what does [course] cover"
- **search_course_content**: For specific concepts, code examples, detailed explanations, or lesson content

Sequential Tool Examples:
- "Search for a course that discusses the same topic as lesson 4 of course X" → get outline for course X → search for similar topics
- "Compare the approach in course A with course B" → search content in course A → search content in course B
- "What advanced topics does the MCP course cover after the basics?" → get course outline → search for advanced content

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool, then provide complete course title, course link, and numbered lesson list
- **Course content questions**: Use search_course_content tool, then answer based on retrieved content
- **Multi-step questions**: Use sequential tool calls to gather all needed information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response with safety check
        if response.content and hasattr(response.content[0], 'text'):
            return response.content[0].text
        else:
            return "Error: No valid response content received from AI."
    
    def generate_response_with_sequential_tools(self, query: str,
                                              conversation_history: Optional[str] = None,
                                              tools: Optional[List] = None,
                                              tool_manager=None,
                                              max_rounds: int = 2) -> str:
        """
        Generate AI response with support for up to 2 sequential tool calls.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context  
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default 2)
            
        Returns:
            Generated response as string
        """
        
        # Initialize round state
        round_state = ToolRoundState(max_rounds=max_rounds)
        
        # Build system content
        round_state.system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize messages with user query
        round_state.messages = [{"role": "user", "content": query}]
        
        try:
            last_response = None
            
            # Execute sequential tool calling rounds
            while round_state.can_continue():
                response = self._execute_tool_round(round_state, tools, tool_manager)
                last_response = response
                
                # If response has tool calls, advance to next round
                if response and self._has_tool_calls(response):
                    round_state.advance_round()
                    continue
                
                # If no tool calls but we have a valid response
                if response and not self._has_tool_calls(response):
                    return self._extract_text_response(response)
                
                # No valid response - error state
                round_state.has_errors = True
                break
            
            # If we've exhausted rounds and the last response had tool calls,
            # make a final call without tools to get the synthesized response
            if last_response and self._has_tool_calls(last_response):
                return self._get_final_response(round_state)
            
            # Otherwise return the last response we got
            return self._extract_text_response(last_response) if last_response else "Error: No response received"
            
        except Exception as e:
            return f"Error during sequential tool execution: {str(e)}"
    
    def _execute_tool_round(self, round_state: ToolRoundState, tools: Optional[List], tool_manager) -> Optional[Any]:
        """Execute a single round of tool calling"""
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": round_state.messages.copy(),
            "system": round_state.system_content
        }
        
        # Add tools if available and we haven't exceeded max rounds
        if tools and round_state.current_round <= round_state.max_rounds:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # If response has tool calls, execute them
        if response.stop_reason == "tool_use" and tool_manager:
            # Add AI's tool use response to messages
            round_state.messages.append({"role": "assistant", "content": response.content})
            
            # Execute all tool calls and collect results
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, 
                            **content_block.input
                        )
                        
                        # Record tool execution in state
                        round_state.add_tool_execution(
                            content_block.name,
                            content_block.input,
                            tool_result
                        )
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        })
                    except Exception:
                        round_state.has_errors = True
                        # Continue to add any successful tool results before erroring
                        break
            
            # Add tool results to messages
            if tool_results:
                round_state.messages.append({"role": "user", "content": tool_results})
        
        return response
    
    def _has_tool_calls(self, response) -> bool:
        """Check if response contains tool calls"""
        if not response or not response.content:
            return False
        return any(block.type == "tool_use" for block in response.content)
    
    def _extract_text_response(self, response) -> str:
        """Extract text content from response"""
        if response and response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
        return "Error: No valid response content received from AI."
    
    def _get_final_response(self, round_state: ToolRoundState) -> str:
        """Get final response without tools"""
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": round_state.messages,
            "system": round_state.system_content
        }
        
        try:
            # Get final response
            final_response = self.client.messages.create(**final_params)
            return self._extract_text_response(final_response)
        except Exception as e:
            return f"Error generating final response: {str(e)}"
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        
        # Return final response with safety check
        if final_response.content and hasattr(final_response.content[0], 'text'):
            return final_response.content[0].text
        else:
            return "Error: No valid response content received from AI after tool execution."