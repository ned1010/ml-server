"""
Ollama Model wrapper with various configurations
Supports different Ollama models with customizable parameters
"""

import requests
import json
from typing import Optional, Dict, Any, List, Generator
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class OllamaConfig(BaseModel):
    """Configuration for Ollama models"""
    model_name: str = "llama2"
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    seed: Optional[int] = None
    num_ctx: int = 2048  # Context window size
    num_predict: int = -1  # Number of tokens to predict (-1 = infinite)
    stop: Optional[List[str]] = None
    stream: bool = False

class OllamaModel:
    """Ollama model wrapper with configuration support"""
    
    # Popular Ollama models
    AVAILABLE_MODELS = {
        "llama2": {
            "name": "llama2",
            "description": "Meta's Llama 2 model",
            "size": "7B",
            "context_length": 4096,
            "family": "llama"
        },
        "llama2:13b": {
            "name": "llama2:13b",
            "description": "Meta's Llama 2 13B model",
            "size": "13B",
            "context_length": 4096,
            "family": "llama"
        },
        "llama2:70b": {
            "name": "llama2:70b",
            "description": "Meta's Llama 2 70B model",
            "size": "70B",
            "context_length": 4096,
            "family": "llama"
        },
        "codellama": {
            "name": "codellama",
            "description": "Code Llama for code generation",
            "size": "7B",
            "context_length": 16384,
            "family": "llama"
        },
        "codellama:13b": {
            "name": "codellama:13b",
            "description": "Code Llama 13B for code generation",
            "size": "13B",
            "context_length": 16384,
            "family": "llama"
        },
        "mistral": {
            "name": "mistral",
            "description": "Mistral 7B model",
            "size": "7B",
            "context_length": 8192,
            "family": "mistral"
        },
        "mixtral": {
            "name": "mixtral",
            "description": "Mixtral 8x7B mixture of experts",
            "size": "8x7B",
            "context_length": 32768,
            "family": "mistral"
        },
        "neural-chat": {
            "name": "neural-chat",
            "description": "Intel's Neural Chat model",
            "size": "7B",
            "context_length": 4096,
            "family": "neural-chat"
        },
        "starcode": {
            "name": "starcode",
            "description": "StarCoder for code generation",
            "size": "7B",
            "context_length": 8192,
            "family": "starcode"
        },
        "vicuna": {
            "name": "vicuna",
            "description": "Vicuna chat model",
            "size": "7B",
            "context_length": 2048,
            "family": "vicuna"
        },
        "orca-mini": {
            "name": "orca-mini",
            "description": "Orca Mini model",
            "size": "3B",
            "context_length": 2048,
            "family": "orca"
        }
    }
    
    def __init__(self, config: OllamaConfig = None, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama model
        
        Args:
            config: Ollama configuration
            base_url: Ollama server URL
        """
        self.config = config or OllamaConfig()
        self.base_url = base_url.rstrip('/')
        
        # Test connection
        try:
            self._test_connection()
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server at {base_url}: {str(e)}")
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
    
    def get_available_models_from_server(self) -> List[Dict[str, Any]]:
        """Get list of models available on the Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Error fetching models from server: {str(e)}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        logger.info(f"Pull status: {data['status']}")
                    if data.get("status") == "success":
                        return True
            
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False
    
    def get_model_info(self, model_key: str = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_key and model_key in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_key]
        
        # Try to find by model name
        for key, model_info in self.AVAILABLE_MODELS.items():
            if model_info["name"] == self.config.model_name:
                return model_info
        
        return {"name": self.config.model_name, "description": "Custom model", "size": "Unknown"}
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from Ollama model
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters to override config
            
        Returns:
            Generated response text
        """
        try:
            # Prepare request data
            data = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": kwargs.get("stream", self.config.stream),
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "top_k": kwargs.get("top_k", self.config.top_k),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                    "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
                    "num_ctx": kwargs.get("num_ctx", self.config.num_ctx),
                    "num_predict": kwargs.get("num_predict", self.config.num_predict),
                }
            }
            
            # Add seed if provided
            if self.config.seed or kwargs.get("seed"):
                data["options"]["seed"] = kwargs.get("seed", self.config.seed)
            
            # Add stop sequences if provided
            if self.config.stop or kwargs.get("stop"):
                data["options"]["stop"] = kwargs.get("stop", self.config.stop)
            
            logger.info(f"Generating response with model: {self.config.model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            
            if self.config.stream or kwargs.get("stream"):
                return response  # Return response object for streaming
            else:
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_stream_response(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate streaming response from Ollama model
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        try:
            kwargs["stream"] = True
            response = self.generate_response(prompt, **kwargs)
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
                        
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise
    
    def chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate chat response from Ollama model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        try:
            # Prepare request data
            data = {
                "model": self.config.model_name,
                "messages": messages,
                "stream": kwargs.get("stream", self.config.stream),
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "top_k": kwargs.get("top_k", self.config.top_k),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                    "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
                    "num_ctx": kwargs.get("num_ctx", self.config.num_ctx),
                    "num_predict": kwargs.get("num_predict", self.config.num_predict),
                }
            }
            
            # Add seed if provided
            if self.config.seed or kwargs.get("seed"):
                data["options"]["seed"] = kwargs.get("seed", self.config.seed)
            
            logger.info(f"Generating chat response with model: {self.config.model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data,
                timeout=300
            )
            response.raise_for_status()
            
            if self.config.stream or kwargs.get("stream"):
                return response  # Return response object for streaming
            else:
                result = response.json()
                return result.get("message", {}).get("content", "")
                
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            raise
    
    def chat_stream_response(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """
        Generate streaming chat response from Ollama model
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        try:
            kwargs["stream"] = True
            response = self.chat_response(messages, **kwargs)
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                    if data.get("done", False):
                        break
                        
        except Exception as e:
            logger.error(f"Error in streaming chat response: {str(e)}")
            raise
    
    def update_config(self, **kwargs):
        """Update model configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    @classmethod
    def create_preset(cls, preset_name: str, base_url: str = "http://localhost:11434") -> 'OllamaModel':
        """
        Create model with preset configurations
        
        Args:
            preset_name: Name of the preset
            base_url: Ollama server URL
            
        Returns:
            Configured OllamaModel instance
        """
        presets = {
            "creative": OllamaConfig(
                model_name="llama2",
                temperature=0.9,
                top_p=0.9,
                repeat_penalty=1.0,
                num_ctx=4096
            ),
            "balanced": OllamaConfig(
                model_name="llama2",
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                num_ctx=4096
            ),
            "precise": OllamaConfig(
                model_name="llama2",
                temperature=0.1,
                top_p=0.1,
                repeat_penalty=1.2,
                num_ctx=4096
            ),
            "code": OllamaConfig(
                model_name="codellama",
                temperature=0.2,
                top_p=0.9,
                repeat_penalty=1.1,
                num_ctx=16384
            ),
            "chat": OllamaConfig(
                model_name="vicuna",
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                num_ctx=2048
            ),
            "fast": OllamaConfig(
                model_name="orca-mini",
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                num_ctx=2048,
                num_predict=512
            )
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        return cls(config=presets[preset_name], base_url=base_url)


# Example usage and factory functions
def create_ollama_model(model_name: str = "llama2", base_url: str = "http://localhost:11434", **config_kwargs) -> OllamaModel:
    """Factory function to create Ollama model with custom config"""
    config = OllamaConfig(model_name=model_name, **config_kwargs)
    return OllamaModel(config=config, base_url=base_url)

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available Ollama models"""
    return OllamaModel.AVAILABLE_MODELS
