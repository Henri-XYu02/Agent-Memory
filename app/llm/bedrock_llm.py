"""
AWS Bedrock LLM wrapper for LangChain integration.
"""

import json
import base64
from typing import Any, Dict, List, Optional, Iterator, AsyncIterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration
import boto3
from botocore.exceptions import ClientError
import os


class BedrockChatModel(BaseChatModel):
    """AWS Bedrock LLM wrapper for LangChain."""
    
    # Define model fields
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    region_name: str = "us-west-2"
    temperature: float = 0.0
    max_tokens: int = 1000
    top_p: float = 0.9
    bedrock_client: Any = None
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-west-2",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        **kwargs: Any,
    ):
        """Initialize the Bedrock LLM.
        
        Args:
            model_id: The Bedrock model ID to use
            region_name: AWS region name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
        """

        super().__init__(**kwargs)
        self.model_id = model_id
        self.region_name = region_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # Initialize Bedrock client
        session_kwargs = {"region_name": region_name}
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        if aws_access_key_id is not None:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if aws_secret_access_key is not None:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        if aws_session_token is not None:
            session_kwargs["aws_session_token"] = aws_session_token

        self.bedrock_client = boto3.client("bedrock-runtime", **session_kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "bedrock"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call the Bedrock model."""
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            if "claude" in self.model_id.lower():
                response_text = self._call_claude(prompt, stop, run_manager, **kwargs)
            elif "llama" in self.model_id.lower():
                response_text = self._call_llama(prompt, stop, run_manager, **kwargs)
            elif "titan" in self.model_id.lower():
                response_text = self._call_titan(prompt, stop, run_manager, **kwargs)
            else:
                # Default to Claude format
                response_text = self._call_claude(prompt, stop, run_manager, **kwargs)
            
            # Create ChatResult
            message = AIMessage(content=response_text)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            raise Exception(f"Bedrock API error ({error_code}): {error_message}")
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to a single prompt string."""
        prompt_parts = []
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            else:
                prompt_parts.append(f"{message.__class__.__name__}: {message.content}")
        
        return "\n\n".join(prompt_parts)
    
    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "BedrockChatModel":
        """Bind tools to the model. For Bedrock, we'll return self as tools are handled by LangGraph."""
        # For Bedrock models, we don't need to bind tools as LangGraph handles tool calling
        # We just return self to maintain compatibility
        return self
    
    def _call_claude(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Claude model via Bedrock."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        if stop:
            body["stop_sequences"] = stop
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
    
    def _call_llama(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Llama model via Bedrock."""
        body = {
            "prompt": prompt,
            "max_gen_len": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        if stop:
            body["stop"] = stop
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["generation"]
    
    def _call_titan(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Titan model via Bedrock."""
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": self.max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            }
        }
        
        if stop:
            body["textGenerationConfig"]["stopSequences"] = stop
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["results"][0]["outputText"]
    


# Predefined model configurations
BEDROCK_MODELS = {
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "llama2-13b": "meta.llama2-13b-chat-v1",
    "llama2-70b": "meta.llama2-70b-chat-v1",
    "llama3-8b": "meta.llama3-8b-instruct-v1:0",
    "llama3-70b": "meta.llama3-70b-instruct-v1:0",
    "titan-text-express": "amazon.titan-text-express-v1",
    "titan-text-lite": "amazon.titan-text-lite-v1",
}


def create_bedrock_llm(
    model_name: str = "claude-3-sonnet",
    region_name: str = "us-east-1",
    **kwargs: Any,
) -> BedrockChatModel:
    """Create a Bedrock LLM instance.
    
    Args:
        model_name: Name of the model (from BEDROCK_MODELS)
        region_name: AWS region name
        **kwargs: Additional arguments for BedrockLLM
        
    Returns:
        BedrockLLM instance
    """
    if model_name not in BEDROCK_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(BEDROCK_MODELS.keys())}")
    
    model_id = BEDROCK_MODELS[model_name]
    return BedrockChatModel(model_id=model_id, region_name=region_name, **kwargs)
