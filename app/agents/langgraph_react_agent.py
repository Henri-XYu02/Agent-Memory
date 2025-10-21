from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.language_models.base import BaseLanguageModel
from typing import List, Dict, Any, Optional, Union
import os

# Import our custom tools
from app.tools.calculator import evaluate_expression
from app.tools.web_search_content import search_and_fetch_content, fetch_web_content
from app.tools.file_reader import read_local_file
from app.tools.reddit_search import search_reddit
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
def search_web_with_content(query: str, max_results: int = 3, content_per_page: int = 3000) -> str:
    """Search the web and get actual content from top results.

    This tool searches the web and automatically fetches the actual text content
    from the top search results, providing you with the information you need
    without having to make separate fetch requests.

    Args:
        query: The search query
        max_results: Maximum number of search results to fetch content from (default: 3)
        content_per_page: Maximum characters to fetch per webpage (default: 3000)

    Returns:
        Formatted string with search results and their actual webpage content
    """
    return search_and_fetch_content(query, max_results, content_per_page)

@tool
def fetch_web_page(url: str, max_chars: int = 5000) -> str:
    """Fetch content from a web page.

    Args:
        url: The URL of the web page to fetch
        max_chars: Maximum number of characters to return (default: 5000)

    Returns:
        The text content of the web page or an error message
    """
    return fetch_web_content(url, max_chars)

@tool
def read_file(file_path: str, max_chars: int = 10000) -> str:
    """Read content from a local file on the system.

    Use this tool to read text files, code files, or any text-based documents.

    Args:
        file_path: Path to the file to read (absolute or relative path)
        max_chars: Maximum number of characters to return (default: 10000)

    Returns:
        The content of the file or an error message if the file cannot be read
    """
    return read_local_file(file_path, max_chars)


@tool
def reddit_search(query: str, subreddit: Optional[str] = None, max_results: int = 5, sort: str = "relevance") -> str:
    """Search Reddit posts and get actual post content.

    Use this to find discussions, opinions, and information from Reddit communities.

    Args:
        query: The search query
        subreddit: Optional subreddit name to search within (e.g., "python", "programming")
        max_results: Maximum number of results to return (default: 5)
        sort: Sort method - "relevance", "hot", "top", "new", "comments" (default: "relevance")

    Returns:
        Formatted string with Reddit posts including titles, scores, authors, and content
    """
    return search_reddit(query, subreddit, max_results, sort)


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
            [calculator, search_web_with_content, read_file, reddit_search, fetch_web_page],
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