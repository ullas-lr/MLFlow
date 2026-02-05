"""
Ollama Model Client
Handles communication with local Ollama server
"""

import time
import requests
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama models"""
    
    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        """
        Initialize Ollama client
        
        Args:
            model: Name of the Ollama model to use
            host: Ollama server URL
        """
        self.model = model
        self.host = host
        self.api_url = f"{host}/api/generate"
        self.chat_url = f"{host}/api/chat"
        
    def check_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stream: Whether to stream response
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    "success": True,
                    "response": data.get("response", ""),
                    "model": self.model,
                    "prompt": prompt,
                    "latency": elapsed_time,
                    "total_duration": data.get("total_duration", 0) / 1e9,  # Convert to seconds
                    "load_duration": data.get("load_duration", 0) / 1e9,
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0) / 1e9,
                }
                
                # Calculate tokens per second
                if result["eval_duration"] > 0:
                    result["tokens_per_second"] = result["eval_count"] / result["eval_duration"]
                else:
                    result["tokens_per_second"] = 0
                
                return result
            else:
                logger.error(f"Error from Ollama: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "latency": elapsed_time,
                    "model": self.model,
                    "prompt": prompt,
                }
                
        except requests.exceptions.Timeout:
            elapsed_time = time.time() - start_time
            logger.error("Request to Ollama timed out")
            return {
                "success": False,
                "error": "Request timeout",
                "latency": elapsed_time,
                "model": self.model,
                "prompt": prompt,
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e),
                "latency": elapsed_time,
                "model": self.model,
                "prompt": prompt,
            }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Chat completion endpoint
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    "success": True,
                    "response": data.get("message", {}).get("content", ""),
                    "model": self.model,
                    "latency": elapsed_time,
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "eval_count": data.get("eval_count", 0),
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "latency": elapsed_time,
                }
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error in chat: {e}")
            return {
                "success": False,
                "error": str(e),
                "latency": elapsed_time,
            }


if __name__ == "__main__":
    # Quick test
    client = OllamaClient()
    
    print("Checking connection...")
    if client.check_connection():
        print("✓ Connected to Ollama")
        
        print("\nAvailable models:")
        models = client.list_models()
        for model in models:
            print(f"  - {model}")
        
        print("\nTesting generation...")
        result = client.generate("What is machine learning?", max_tokens=100)
        
        if result["success"]:
            print(f"\n✓ Response: {result['response'][:100]}...")
            print(f"  Latency: {result['latency']:.2f}s")
            print(f"  Tokens/sec: {result.get('tokens_per_second', 0):.2f}")
        else:
            print(f"\n✗ Error: {result['error']}")
    else:
        print("✗ Could not connect to Ollama")
        print("\nMake sure Ollama is running:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull a model: ollama pull llama2")
