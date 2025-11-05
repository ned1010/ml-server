"""
Model configuration management
Handles loading and saving model configurations
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from model_interface import ModelFactory, UnifiedModelInterface

logger = logging.getLogger(__name__)

class ModelConfigManager:
    """Manages model configurations"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.default_configs = {
            "openai_default": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "openai_creative": {
                "provider": "openai",
                "model_name": "gpt-4-turbo-preview",
                "temperature": 0.9,
                "max_tokens": 2000,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            },
            "openai_precise": {
                "provider": "openai",
                "model_name": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 1000,
                "top_p": 0.1,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "openai_fast": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": True
            },
            "ollama_default": {
                "provider": "ollama",
                "model_name": "llama2",
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
                "base_url": "http://localhost:11434"
            },
            "ollama_creative": {
                "provider": "ollama",
                "model_name": "llama2",
                "temperature": 0.9,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.0,
                "num_ctx": 4096,
                "base_url": "http://localhost:11434"
            },
            "ollama_precise": {
                "provider": "ollama",
                "model_name": "llama2",
                "temperature": 0.1,
                "top_k": 10,
                "top_p": 0.1,
                "repeat_penalty": 1.2,
                "num_ctx": 4096,
                "base_url": "http://localhost:11434"
            },
            "ollama_code": {
                "provider": "ollama",
                "model_name": "codellama",
                "temperature": 0.2,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 16384,
                "base_url": "http://localhost:11434"
            },
            "ollama_chat": {
                "provider": "ollama",
                "model_name": "vicuna",
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
                "base_url": "http://localhost:11434"
            }
        }
    
    def save_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Save configuration to file
        
        Args:
            name: Configuration name
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = self.config_dir / f"{name}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration '{name}' to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration '{name}': {str(e)}")
            return False
    
    def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from file
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration dictionary or None if not found
        """
        # Check if it's a default configuration
        if name in self.default_configs:
            return self.default_configs[name].copy()
        
        # Try to load from file
        config_path = self.config_dir / f"{name}.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration '{name}' from {config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration '{name}': {str(e)}")
                return None
        
        logger.warning(f"Configuration '{name}' not found")
        return None
    
    def list_configs(self) -> Dict[str, str]:
        """
        List all available configurations
        
        Returns:
            Dictionary mapping config names to their sources
        """
        configs = {}
        
        # Add default configurations
        for name in self.default_configs:
            configs[name] = "default"
        
        # Add file-based configurations
        for config_file in self.config_dir.glob("*.json"):
            name = config_file.stem
            if name not in configs:  # Don't override defaults
                configs[name] = "file"
        
        return configs
    
    def delete_config(self, name: str) -> bool:
        """
        Delete a configuration file
        
        Args:
            name: Configuration name
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.default_configs:
            logger.warning(f"Cannot delete default configuration '{name}'")
            return False
        
        config_path = self.config_dir / f"{name}.json"
        if config_path.exists():
            try:
                config_path.unlink()
                logger.info(f"Deleted configuration '{name}'")
                return True
            except Exception as e:
                logger.error(f"Error deleting configuration '{name}': {str(e)}")
                return False
        
        logger.warning(f"Configuration '{name}' not found")
        return False
    
    def create_model_from_config(self, name: str) -> Optional[UnifiedModelInterface]:
        """
        Create model instance from configuration
        
        Args:
            name: Configuration name
            
        Returns:
            UnifiedModelInterface instance or None if failed
        """
        config = self.load_config(name)
        if config is None:
            return None
        
        try:
            return ModelFactory.create_model_from_config(config)
        except Exception as e:
            logger.error(f"Error creating model from config '{name}': {str(e)}")
            return None
    
    def save_defaults(self):
        """Save all default configurations to files"""
        for name, config in self.default_configs.items():
            self.save_config(name, config)
        logger.info("Saved all default configurations")
    
    def get_config_template(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration template for a provider
        
        Args:
            provider: Provider name ("openai" or "ollama")
            
        Returns:
            Configuration template
        """
        if provider.lower() == "openai":
            return {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stop": None,
                "stream": False,
                "api_key": None  # Will use environment variable if None
            }
        elif provider.lower() == "ollama":
            return {
                "provider": "ollama",
                "model_name": "llama2",
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "seed": None,
                "num_ctx": 2048,
                "num_predict": -1,
                "stop": None,
                "stream": False,
                "base_url": "http://localhost:11434"
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

# Global configuration manager instance
config_manager = ModelConfigManager()

# Convenience functions
def save_model_config(name: str, config: Dict[str, Any]) -> bool:
    """Save model configuration"""
    return config_manager.save_config(name, config)

def load_model_config(name: str) -> Optional[Dict[str, Any]]:
    """Load model configuration"""
    return config_manager.load_config(name)

def create_model_from_config(name: str) -> Optional[UnifiedModelInterface]:
    """Create model from configuration"""
    return config_manager.create_model_from_config(name)

def list_model_configs() -> Dict[str, str]:
    """List all available model configurations"""
    return config_manager.list_configs()

def get_config_template(provider: str) -> Dict[str, Any]:
    """Get configuration template for provider"""
    return config_manager.get_config_template(provider)

# Example usage
if __name__ == "__main__":
    # List all configurations
    print("Available configurations:")
    configs = list_model_configs()
    for name, source in configs.items():
        print(f"  - {name} ({source})")
    
    # Create models from configurations
    print("\nTesting configurations:")
    test_configs = ["openai_default", "ollama_default"]
    
    for config_name in test_configs:
        try:
            model = create_model_from_config(config_name)
            if model:
                info = model.get_model_info()
                print(f"  ✓ {config_name}: {info['name']} ({info.get('provider', 'unknown')})")
            else:
                print(f"  ✗ {config_name}: Failed to create model")
        except Exception as e:
            print(f"  ✗ {config_name}: {str(e)}")
    
    # Save default configurations to files
    config_manager.save_defaults()
    print("\nDefault configurations saved to files")
