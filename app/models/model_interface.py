"""
Unified model interface for OpenAI and Ollama models
Provides a consistent API for different model providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generator, Union
from enum import Enum
import logging

from .openai_model import OpenAIModel, OpenAIConfig
from .ollama_model import OllamaModel, OllamaConfig

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"

class BaseModelInterface(ABC):
    """Abstract base class for model interfaces"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the model"""
        pass
    
    @abstractmethod
    def generate_stream_response(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Generate a streaming response from the model"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        pass
    
    @abstractmethod
    def update_config(self, **kwargs):
        """Update model configuration"""
        pass

class UnifiedModelInterface(BaseModelInterface):
    """Unified interface for different model providers"""
    
    def __init__(self, provider: ModelProvider, model_instance: Union[OpenAIModel, OllamaModel]):
        """
        Initialize unified model interface
        
        Args:
            provider: Model provider type
            model_instance: Instance of the model (OpenAI or Ollama)
        """
        self.provider = provider
        self.model = model_instance
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using the appropriate model
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        try:
            if self.provider == ModelProvider.OPENAI:
                return self.model.generate_response(messages, **kwargs)
            elif self.provider == ModelProvider.OLLAMA:
                # Convert messages to prompt for Ollama if using generate_response
                if hasattr(self.model, 'chat_response'):
                    return self.model.chat_response(messages, **kwargs)
                else:
                    # Fallback to prompt-based generation
                    prompt = self._messages_to_prompt(messages)
                    return self.model.generate_response(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating response with {self.provider.value}: {str(e)}")
            raise
    
    def generate_stream_response(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """
        Generate streaming response using the appropriate model
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        try:
            if self.provider == ModelProvider.OPENAI:
                yield from self.model.generate_stream_response(messages, **kwargs)
            elif self.provider == ModelProvider.OLLAMA:
                # Use chat streaming if available
                if hasattr(self.model, 'chat_stream_response'):
                    yield from self.model.chat_stream_response(messages, **kwargs)
                else:
                    # Fallback to prompt-based streaming
                    prompt = self._messages_to_prompt(messages)
                    yield from self.model.generate_stream_response(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating streaming response with {self.provider.value}: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = self.model.get_model_info()
        info["provider"] = self.provider.value
        return info
    
    def update_config(self, **kwargs):
        """Update model configuration"""
        self.model.update_config(**kwargs)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert messages to a single prompt string for models that don't support chat format
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

class ModelFactory:
    """Factory class for creating model instances"""
    
    @staticmethod
    def create_openai_model(
        model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        preset: str = None,
        **config_kwargs
    ) -> UnifiedModelInterface:
        """
        Create OpenAI model instance
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key
            preset: Preset configuration name
            **config_kwargs: Additional configuration parameters
            
        Returns:
            UnifiedModelInterface instance
        """
        if preset:
            model = OpenAIModel.create_preset(preset, api_key=api_key)
        else:
            config = OpenAIConfig(model_name=model_name, **config_kwargs)
            model = OpenAIModel(config=config, api_key=api_key)
        
        return UnifiedModelInterface(ModelProvider.OPENAI, model)
    
    @staticmethod
    def create_ollama_model(
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        preset: str = None,
        **config_kwargs
    ) -> UnifiedModelInterface:
        """
        Create Ollama model instance
        
        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
            preset: Preset configuration name
            **config_kwargs: Additional configuration parameters
            
        Returns:
            UnifiedModelInterface instance
        """
        if preset:
            model = OllamaModel.create_preset(preset, base_url=base_url)
        else:
            config = OllamaConfig(model_name=model_name, **config_kwargs)
            model = OllamaModel(config=config, base_url=base_url)
        
        return UnifiedModelInterface(ModelProvider.OLLAMA, model)
    
    @staticmethod
    def create_model_from_config(config: Dict[str, Any]) -> UnifiedModelInterface:
        """
        Create model instance from configuration dictionary
        
        Args:
            config: Configuration dictionary
            
        Returns:
            UnifiedModelInterface instance
        """
        provider = config.get("provider", "openai").lower()
        
        if provider == "openai":
            return ModelFactory.create_openai_model(
                model_name=config.get("model_name", "gpt-3.5-turbo"),
                api_key=config.get("api_key"),
                preset=config.get("preset"),
                **{k: v for k, v in config.items() if k not in ["provider", "model_name", "api_key", "preset"]}
            )
        elif provider == "ollama":
            return ModelFactory.create_ollama_model(
                model_name=config.get("model_name", "llama2"),
                base_url=config.get("base_url", "http://localhost:11434"),
                preset=config.get("preset"),
                **{k: v for k, v in config.items() if k not in ["provider", "model_name", "base_url", "preset"]}
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

# Convenience functions
def create_model(
    provider: str,
    model_name: str = None,
    preset: str = None,
    **kwargs
) -> UnifiedModelInterface:
    """
    Convenience function to create a model
    
    Args:
        provider: Model provider ("openai" or "ollama")
        model_name: Model name
        preset: Preset configuration name
        **kwargs: Additional configuration
        
    Returns:
        UnifiedModelInterface instance
    """
    if provider.lower() == "openai":
        return ModelFactory.create_openai_model(
            model_name=model_name or "gpt-3.5-turbo",
            preset=preset,
            **kwargs
        )
    elif provider.lower() == "ollama":
        return ModelFactory.create_ollama_model(
            model_name=model_name or "llama2",
            preset=preset,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def get_available_models(provider: str = None) -> Dict[str, Any]:
    """
    Get available models for a provider or all providers
    
    Args:
        provider: Specific provider to get models for
        
    Returns:
        Dictionary of available models
    """
    if provider:
        if provider.lower() == "openai":
            return {"openai": OpenAIModel.AVAILABLE_MODELS}
        elif provider.lower() == "ollama":
            return {"ollama": OllamaModel.AVAILABLE_MODELS}
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    else:
        return {
            "openai": OpenAIModel.AVAILABLE_MODELS,
            "ollama": OllamaModel.AVAILABLE_MODELS
        }
