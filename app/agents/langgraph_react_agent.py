from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.language_models.base import BaseLanguageModel
from typing import List, Dict, Any, Optional, Union
import os

# Import our custom tools
from app.tools.calculator import evaluate_expression
from app.tools.web_search import web_search
from app.llm.bedrock_llm import BedrockChatModel, create_bedrock_llm


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The result of the mathematical expression or an error message
    """
    result = evaluate_expression(expression)
    return str(result)


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        A formatted string with search results
    """
    results = web_search(query, max_results)
    if not results:
        return "No search results found."
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result['title']}\n   URL: {result['url']}")
    
    return "\n\n".join(formatted_results)


class LangGraphReActAgent:
    def __init__(
        self, 
        model_name: str = "claude-3-opus", 
        use_memory: bool = False, 
        memory_service: Optional[Any] = None,
        provider: str = "bedrock",
        **model_kwargs: Any
    ):
        """Initialize the LangGraph ReAct agent.
        
        Args:
            model_name: The model to use (OpenAI model name or Bedrock model name)
            use_memory: Whether to use memory/checkpointing
            memory_service: Optional mem0-based memory service
            provider: Model provider ("openai" or "bedrock")
            **model_kwargs: Additional arguments for the model
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.use_memory = use_memory
        self.memory_service = memory_service
        
        # Initialize the appropriate model
        if self.provider == "openai":
            self.model = ChatOpenAI(model=model_name, **model_kwargs)
        elif self.provider == "bedrock":
            self.model = create_bedrock_llm(model_name=model_name, **model_kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'bedrock'")
        
        # Initialize checkpointer for memory
        if use_memory:
            self.checkpointer = InMemorySaver()
        else:
            self.checkpointer = None
        
        # Create the ReAct agent with our tools
        self.agent = create_react_agent(
            self.model,
            [calculator, search_web],
            checkpointer=self.checkpointer,
        )
    
    def run(self, message: str, thread_id: str = "default", system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent with a message.
        
        Args:
            message: The user message
            thread_id: Unique identifier for the conversation thread
            system_prompt: Optional system prompt to override default behavior
            
        Returns:
            The agent's response
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", message))
        
        # Get response from agent
        response = self.agent.invoke({"messages": messages}, config=config)
        
        
        return response


# For backward compatibility and direct usage
def create_agent(
    model_name: str = "claude-3-opus", 
    use_memory: bool = False, 
    provider: str = "bedrock",
    **model_kwargs: Any
) -> LangGraphReActAgent:
    """Create a LangGraph ReAct agent instance.
    
    Args:
        model_name: The model to use
        use_memory: Whether to use memory/checkpointing
        provider: Model provider ("openai" or "bedrock")
        **model_kwargs: Additional arguments for the model
    """
    return LangGraphReActAgent(
        model_name=model_name, 
        use_memory=use_memory, 
        provider=provider,
        **model_kwargs
    )


def run_agent(
    user_id: str, 
    session_id: Optional[str], 
    message: str, 
    system_prompt: Optional[str] = None,
    provider: str = "bedrock",
    model_name: str = "claude-3-opus",
    **model_kwargs: Any
) -> Dict[str, Any]:
    """Run the agent for a specific user and session.
    
    This function is used by the FastAPI endpoint.
    
    Args:
        user_id: User identifier
        session_id: Session identifier
        message: User message
        system_prompt: Optional system prompt
        provider: Model provider ("openai" or "bedrock")
        model_name: Model name
        **model_kwargs: Additional model arguments
    """
    thread_id = f"{user_id}_{session_id}" if session_id else user_id
    agent = create_agent(
        model_name=model_name,
        provider=provider,
        **model_kwargs
    )
    return agent.run(message, thread_id, system_prompt)