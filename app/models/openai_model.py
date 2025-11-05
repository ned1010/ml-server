"""
OpenAI Model wrapper with various configurations
Supports different OpenAI models with customizable parameters
"""

import os
from typing import Optional, Dict, Any, List
from openai import OpenAI
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI models"""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False

class OpenAIModel:
    """OpenAI model wrapper with configuration support"""
    
    # Available OpenAI models
    AVAILABLE_MODELS = {
        "gpt-4": {
            "name": "gpt-4",
            "description": "Most capable GPT-4 model",
            "max_tokens": 8192,
            "cost_per_1k_tokens": {"input": 0.03, "output": 0.06}
        },
        "gpt-4-turbo": {
            "name": "gpt-4-turbo-preview",
            "description": "Latest GPT-4 Turbo model",
            "max_tokens": 128000,
            "cost_per_1k_tokens": {"input": 0.01, "output": 0.03}
        },
        "gpt-3.5-turbo": {
            "name": "gpt-3.5-turbo",
            "description": "Fast and efficient GPT-3.5 model",
            "max_tokens": 4096,
            "cost_per_1k_tokens": {"input": 0.0015, "output": 0.002}
        },
        "gpt-3.5-turbo-16k": {
            "name": "gpt-3.5-turbo-16k",
            "description": "GPT-3.5 with 16k context window",
            "max_tokens": 16384,
            "cost_per_1k_tokens": {"input": 0.003, "output": 0.004}
        }
    }
    
    def __init__(self, config: OpenAIConfig = None, api_key: str = None):
        """
        Initialize OpenAI model
        
        Args:
            config: OpenAI configuration
            api_key: OpenAI API key (if not provided, will use env variable)
        """
        self.config = config or OpenAIConfig()
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Validate model
        if self.config.model_name not in [model["name"] for model in self.AVAILABLE_MODELS.values()]:
            logger.warning(f"Model {self.config.model_name} not in predefined list. Using anyway.")
    
    def get_model_info(self, model_key: str = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_key and model_key in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_key]
        
        # Try to find by model name
        for key, model_info in self.AVAILABLE_MODELS.items():
            if model_info["name"] == self.config.model_name:
                return model_info
        
        return {"name": self.config.model_name, "description": "Custom model", "max_tokens": None}
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response from OpenAI model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters to override config
            
        Returns:
            Generated response text
        """
        try:
            # Merge config with kwargs
            params = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
                "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
                "stream": kwargs.get("stream", self.config.stream),
            }
            
            # Add stop sequences if provided
            if self.config.stop or kwargs.get("stop"):
                params["stop"] = kwargs.get("stop", self.config.stop)
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            logger.info(f"Generating response with model: {self.config.model_name}")
            
            response = self.client.chat.completions.create(**params)
            
            if self.config.stream:
                return response  # Return stream object for streaming
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_stream_response(self, messages: List[Dict[str, str]], **kwargs):
        """
        Generate streaming response from OpenAI model
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        try:
            kwargs["stream"] = True
            response = self.generate_response(messages, **kwargs)
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation)
        For accurate counting, use tiktoken library
        """
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for the request
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        model_info = self.get_model_info()
        if "cost_per_1k_tokens" not in model_info:
            return 0.0
        
        costs = model_info["cost_per_1k_tokens"]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def update_config(self, **kwargs):
        """Update model configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    @classmethod
    def create_preset(cls, preset_name: str, api_key: str = None) -> 'OpenAIModel':
        """
        Create model with preset configurations
        
        Args:
            preset_name: Name of the preset
            api_key: OpenAI API key
            
        Returns:
            Configured OpenAIModel instance
        """
        presets = {
            "creative": OpenAIConfig(
                model_name="gpt-4-turbo-preview",
                temperature=0.9,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            ),
            "balanced": OpenAIConfig(
                model_name="gpt-4-turbo-preview",
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ),
            "precise": OpenAIConfig(
                model_name="gpt-4",
                temperature=0.1,
                top_p=0.1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ),
            "fast": OpenAIConfig(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            ),
            "long_context": OpenAIConfig(
                model_name="gpt-3.5-turbo-16k",
                temperature=0.7,
                max_tokens=8000
            )
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        return cls(config=presets[preset_name], api_key=api_key)


# Example usage and factory functions
def create_openai_model(model_name: str = "gpt-3.5-turbo", **config_kwargs) -> OpenAIModel:
    """Factory function to create OpenAI model with custom config"""
    config = OpenAIConfig(model_name=model_name, **config_kwargs)
    return OpenAIModel(config=config)

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available OpenAI models"""
    return OpenAIModel.AVAILABLE_MODELS