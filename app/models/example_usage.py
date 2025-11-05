"""
Example usage of OpenAI and Ollama models
Demonstrates various configurations and use cases
"""

import asyncio
import os
from typing import List, Dict

from model_interface import ModelFactory, create_model, get_available_models
from openai_model import OpenAIModel, OpenAIConfig
from ollama_model import OllamaModel, OllamaConfig

def example_openai_usage():
    """Example usage of OpenAI models"""
    print("=== OpenAI Model Examples ===")
    
    # Method 1: Using factory with preset
    try:
        model = ModelFactory.create_openai_model(preset="balanced")
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
        
        response = model.generate_response(messages)
        print(f"Preset model response: {response[:100]}...")
        
    except Exception as e:
        print(f"OpenAI error (check API key): {e}")
    
    # Method 2: Using factory with custom config
    try:
        model = ModelFactory.create_openai_model(
            model_name="gpt-3.5-turbo",
            temperature=0.9,
            max_tokens=150
        )
        
        messages = [
            {"role": "user", "content": "Write a creative short story about a robot."}
        ]
        
        response = model.generate_response(messages)
        print(f"Custom config response: {response[:100]}...")
        
    except Exception as e:
        print(f"OpenAI error: {e}")
    
    # Method 3: Direct model creation
    try:
        config = OpenAIConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=100
        )
        openai_model = OpenAIModel(config=config)
        
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = openai_model.generate_response(messages)
        print(f"Direct model response: {response}")
        
        # Get model info
        info = openai_model.get_model_info()
        print(f"Model info: {info}")
        
        # Estimate cost
        input_tokens = openai_model.count_tokens("What is 2+2?")
        output_tokens = openai_model.count_tokens(response)
        cost = openai_model.estimate_cost(input_tokens, output_tokens)
        print(f"Estimated cost: ${cost:.6f}")
        
    except Exception as e:
        print(f"OpenAI error: {e}")

def example_ollama_usage():
    """Example usage of Ollama models"""
    print("\n=== Ollama Model Examples ===")
    
    # Method 1: Using factory with preset
    try:
        model = ModelFactory.create_ollama_model(preset="balanced")
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ]
        
        response = model.generate_response(messages)
        print(f"Preset model response: {response[:100]}...")
        
    except Exception as e:
        print(f"Ollama error (check if server is running): {e}")
    
    # Method 2: Using factory with custom config
    try:
        model = ModelFactory.create_ollama_model(
            model_name="llama2",
            temperature=0.8,
            num_ctx=2048
        )
        
        messages = [
            {"role": "user", "content": "Explain the concept of recursion."}
        ]
        
        response = model.generate_response(messages)
        print(f"Custom config response: {response[:100]}...")
        
    except Exception as e:
        print(f"Ollama error: {e}")
    
    # Method 3: Direct model creation
    try:
        config = OllamaConfig(
            model_name="llama2",
            temperature=0.7,
            num_ctx=4096,
            num_predict=200
        )
        ollama_model = OllamaModel(config=config)
        
        # Check available models on server
        server_models = ollama_model.get_available_models_from_server()
        print(f"Models on server: {[m['name'] for m in server_models]}")
        
        # Generate response
        prompt = "Write a haiku about programming."
        response = ollama_model.generate_response(prompt)
        print(f"Direct model response: {response}")
        
        # Get model info
        info = ollama_model.get_model_info()
        print(f"Model info: {info}")
        
    except Exception as e:
        print(f"Ollama error: {e}")

def example_streaming_usage():
    """Example of streaming responses"""
    print("\n=== Streaming Examples ===")
    
    # OpenAI streaming
    try:
        model = ModelFactory.create_openai_model(preset="fast")
        
        messages = [
            {"role": "user", "content": "Count from 1 to 10 with explanations."}
        ]
        
        print("OpenAI streaming response:")
        for chunk in model.generate_stream_response(messages):
            print(chunk, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"OpenAI streaming error: {e}")
    
    # Ollama streaming
    try:
        model = ModelFactory.create_ollama_model(preset="fast")
        
        messages = [
            {"role": "user", "content": "Tell me a short joke."}
        ]
        
        print("Ollama streaming response:")
        for chunk in model.generate_stream_response(messages):
            print(chunk, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"Ollama streaming error: {e}")

def example_model_comparison():
    """Compare responses from different models"""
    print("\n=== Model Comparison ===")
    
    question = "What are the benefits of renewable energy?"
    messages = [{"role": "user", "content": question}]
    
    models = [
        ("OpenAI GPT-3.5 (Precise)", "openai", {"preset": "precise"}),
        ("OpenAI GPT-3.5 (Creative)", "openai", {"preset": "creative"}),
        ("Ollama Llama2 (Balanced)", "ollama", {"preset": "balanced"}),
        ("Ollama Llama2 (Precise)", "ollama", {"preset": "precise"}),
    ]
    
    for name, provider, config in models:
        try:
            model = create_model(provider, **config)
            response = model.generate_response(messages, max_tokens=100)
            print(f"\n{name}:")
            print(f"{response[:150]}...")
            
        except Exception as e:
            print(f"\n{name}: Error - {e}")

def example_configuration_updates():
    """Example of updating model configurations"""
    print("\n=== Configuration Updates ===")
    
    try:
        # Create model with initial config
        model = create_model("openai", model_name="gpt-3.5-turbo", temperature=0.1)
        
        messages = [{"role": "user", "content": "Write a creative sentence about cats."}]
        
        print("Response with low temperature (0.1):")
        response1 = model.generate_response(messages)
        print(response1)
        
        # Update configuration
        model.update_config(temperature=0.9)
        
        print("\nResponse with high temperature (0.9):")
        response2 = model.generate_response(messages)
        print(response2)
        
    except Exception as e:
        print(f"Configuration update error: {e}")

def show_available_models():
    """Show all available models"""
    print("\n=== Available Models ===")
    
    models = get_available_models()
    
    for provider, provider_models in models.items():
        print(f"\n{provider.upper()} Models:")
        for key, info in provider_models.items():
            print(f"  - {key}: {info['description']}")
            if 'size' in info:
                print(f"    Size: {info['size']}")
            if 'context_length' in info:
                print(f"    Context: {info['context_length']} tokens")

def main():
    """Run all examples"""
    print("Model Examples and Demonstrations")
    print("=" * 50)
    
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Show available models
    show_available_models()
    
    # Run examples
    example_openai_usage()
    example_ollama_usage()
    example_streaming_usage()
    example_model_comparison()
    example_configuration_updates()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNotes:")
    print("- For OpenAI: Set OPENAI_API_KEY environment variable")
    print("- For Ollama: Ensure Ollama server is running (ollama serve)")
    print("- Install required models: ollama pull llama2")

if __name__ == "__main__":
    main()
