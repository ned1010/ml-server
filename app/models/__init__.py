"""
Models package for OpenAI and Ollama integration
Provides unified interface for different LLM providers
"""

from .openai_model import OpenAIModel, OpenAIConfig, create_openai_model
from .ollama_model import OllamaModel, OllamaConfig, create_ollama_model
from .model_interface import (
    UnifiedModelInterface, 
    ModelFactory, 
    ModelProvider,
    create_model,
    get_available_models
)
from .model_config import (
    ModelConfigManager,
    config_manager,
    save_model_config,
    load_model_config,
    create_model_from_config,
    list_model_configs,
    get_config_template
)

__version__ = "1.0.0"
__author__ = "Intreli Team"

# Main exports
__all__ = [
    # OpenAI
    "OpenAIModel",
    "OpenAIConfig", 
    "create_openai_model",
    
    # Ollama
    "OllamaModel",
    "OllamaConfig",
    "create_ollama_model",
    
    # Unified Interface
    "UnifiedModelInterface",
    "ModelFactory",
    "ModelProvider",
    "create_model",
    "get_available_models",
    
    # Configuration Management
    "ModelConfigManager",
    "config_manager",
    "save_model_config",
    "load_model_config", 
    "create_model_from_config",
    "list_model_configs",
    "get_config_template"
]

# Quick start examples
QUICK_START_EXAMPLES = {
    "openai_simple": {
        "description": "Simple OpenAI GPT-3.5 model",
        "code": """
from models import create_model

model = create_model("openai", model_name="gpt-3.5-turbo")
messages = [{"role": "user", "content": "Hello!"}]
response = model.generate_response(messages)
print(response)
        """
    },
    "ollama_simple": {
        "description": "Simple Ollama Llama2 model",
        "code": """
from models import create_model

model = create_model("ollama", model_name="llama2")
messages = [{"role": "user", "content": "Hello!"}]
response = model.generate_response(messages)
print(response)
        """
    },
    "config_based": {
        "description": "Using configuration-based model creation",
        "code": """
from models import create_model_from_config

# Use predefined configuration
model = create_model_from_config("openai_creative")
messages = [{"role": "user", "content": "Write a poem"}]
response = model.generate_response(messages)
print(response)
        """
    },
    "streaming": {
        "description": "Streaming response example",
        "code": """
from models import create_model

model = create_model("openai", preset="fast")
messages = [{"role": "user", "content": "Count to 10"}]

for chunk in model.generate_stream_response(messages):
    print(chunk, end="", flush=True)
        """
    }
}

def print_quick_start():
    """Print quick start examples"""
    print("Models Package - Quick Start Examples")
    print("=" * 50)
    
    for name, example in QUICK_START_EXAMPLES.items():
        print(f"\n{name.upper()}: {example['description']}")
        print("-" * 30)
        print(example['code'].strip())
    
    print("\n" + "=" * 50)
    print("Available configurations:")
    configs = list_model_configs()
    for config_name, source in configs.items():
        print(f"  - {config_name} ({source})")

def get_model_recommendations():
    """Get model recommendations for different use cases"""
    return {
        "general_chat": {
            "openai": "openai_default",
            "ollama": "ollama_chat",
            "description": "General purpose conversational AI"
        },
        "creative_writing": {
            "openai": "openai_creative", 
            "ollama": "ollama_creative",
            "description": "Creative content generation with higher temperature"
        },
        "code_generation": {
            "openai": "openai_precise",
            "ollama": "ollama_code", 
            "description": "Code generation and programming assistance"
        },
        "fast_responses": {
            "openai": "openai_fast",
            "ollama": "ollama_default",
            "description": "Quick responses with lower token limits"
        },
        "analytical_tasks": {
            "openai": "openai_precise",
            "ollama": "ollama_precise", 
            "description": "Analytical and factual tasks requiring precision"
        }
    }

# Package information
def get_package_info():
    """Get package information"""
    return {
        "name": "models",
        "version": __version__,
        "description": "Unified interface for OpenAI and Ollama models",
        "supported_providers": ["OpenAI", "Ollama"],
        "features": [
            "Unified API for multiple providers",
            "Configuration management",
            "Streaming support", 
            "Preset configurations",
            "Cost estimation (OpenAI)",
            "Model information and metadata"
        ]
    }
