"""
LLM Client - Unified interface for multiple LLM providers

Supports:
- Ollama (local deployment)
- OpenAI API
- Azure OpenAI
- Local models (LM Studio, etc.)

Usage:
    from llm_client import LLMClient
    
    client = LLMClient()  # Auto-loads from config
    response = client.chat("What is the capital of France?")
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger("LLMClient")

# Try to import requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed. Run: pip install requests")

# Import config loader
try:
    from config_loader import get_config
    _config = get_config()
except ImportError:
    _config = None


def _get_cfg(key: str, default: Any = None) -> Any:
    """Get configuration value with fallback."""
    if _config is not None:
        return _config.get(key, default)
    return default


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM fails."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid."""
    pass


# =============================================================================
# Abstract Base Provider
# =============================================================================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.1,
             max_tokens: int = 1024) -> str:
        """Send chat messages and get response."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


# =============================================================================
# Ollama Provider (Local)
# =============================================================================

class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, 
                 base_url: str = None, 
                 model: str = None,
                 timeout: int = None):
        self.base_url = base_url or _get_cfg("llm.ollama.base_url", "http://localhost:11434")
        self.model = model or _get_cfg("llm.ollama.model", "qwen2.5:7b")
        self.timeout = timeout or _get_cfg("llm.ollama.timeout", 60)
        
        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        if not HAS_REQUESTS:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = None,
             max_tokens: int = None) -> str:
        """Send chat request to Ollama."""
        if not HAS_REQUESTS:
            raise LLMConnectionError("requests library not installed")
        
        temperature = temperature if temperature is not None else _get_cfg("llm.ollama.temperature", 0.1)
        max_tokens = max_tokens if max_tokens is not None else _get_cfg("llm.ollama.max_tokens", 1024)
        
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            raise LLMConnectionError(f"Cannot connect to Ollama at {self.base_url}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError(f"Ollama request timed out after {self.timeout}s")
        except Exception as e:
            raise LLMError(f"Ollama error: {e}")


# =============================================================================
# OpenAI Provider
# =============================================================================

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self,
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None,
                 timeout: int = None):
        self.api_key = api_key or _get_cfg("llm.openai.api_key", "")
        self.base_url = base_url or _get_cfg("llm.openai.base_url", "https://api.openai.com/v1")
        self.model = model or _get_cfg("llm.openai.model", "gpt-4o-mini")
        self.timeout = timeout or _get_cfg("llm.openai.timeout", 30)
        
        self.base_url = self.base_url.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key) and self.api_key != "${OPENAI_API_KEY}"
    
    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = None,
             max_tokens: int = None) -> str:
        """Send chat request to OpenAI."""
        if not HAS_REQUESTS:
            raise LLMConnectionError("requests library not installed")
        
        if not self.is_available():
            raise LLMConnectionError("OpenAI API key not configured")
        
        temperature = temperature if temperature is not None else _get_cfg("llm.openai.temperature", 0.1)
        max_tokens = max_tokens if max_tokens is not None else _get_cfg("llm.openai.max_tokens", 1024)
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise LLMConnectionError(f"Cannot connect to OpenAI at {self.base_url}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError(f"OpenAI request timed out after {self.timeout}s")
        except KeyError:
            raise LLMResponseError(f"Invalid OpenAI response format")
        except Exception as e:
            raise LLMError(f"OpenAI error: {e}")


# =============================================================================
# Azure OpenAI Provider
# =============================================================================

class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI API provider."""
    
    def __init__(self,
                 api_key: str = None,
                 endpoint: str = None,
                 deployment_name: str = None,
                 api_version: str = None,
                 timeout: int = None):
        self.api_key = api_key or _get_cfg("llm.azure.api_key", "")
        self.endpoint = endpoint or _get_cfg("llm.azure.endpoint", "")
        self.deployment_name = deployment_name or _get_cfg("llm.azure.deployment_name", "gpt-4o-mini")
        self.api_version = api_version or _get_cfg("llm.azure.api_version", "2024-02-15-preview")
        self.timeout = timeout or _get_cfg("llm.openai.timeout", 30)
        
        self.endpoint = self.endpoint.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if Azure credentials are configured."""
        return (bool(self.api_key) and self.api_key != "${AZURE_OPENAI_API_KEY}" and
                bool(self.endpoint) and self.endpoint != "${AZURE_OPENAI_ENDPOINT}")
    
    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = None,
             max_tokens: int = None) -> str:
        """Send chat request to Azure OpenAI."""
        if not HAS_REQUESTS:
            raise LLMConnectionError("requests library not installed")
        
        if not self.is_available():
            raise LLMConnectionError("Azure OpenAI credentials not configured")
        
        temperature = temperature if temperature is not None else 0.1
        max_tokens = max_tokens if max_tokens is not None else 1024
        
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise LLMConnectionError(f"Cannot connect to Azure at {self.endpoint}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError(f"Azure request timed out after {self.timeout}s")
        except KeyError:
            raise LLMResponseError(f"Invalid Azure response format")
        except Exception as e:
            raise LLMError(f"Azure error: {e}")


# =============================================================================
# Local Model Provider (LM Studio, etc.)
# =============================================================================

class LocalModelProvider(BaseLLMProvider):
    """Local model provider (LM Studio, text-generation-webui, etc.)."""
    
    def __init__(self,
                 base_url: str = None,
                 model: str = None,
                 timeout: int = None):
        self.base_url = base_url or _get_cfg("llm.local.base_url", "http://localhost:1234/v1")
        self.model = model or _get_cfg("llm.local.model", "local-model")
        self.timeout = timeout or _get_cfg("llm.local.timeout", 120)
        
        self.base_url = self.base_url.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if local server is running."""
        if not HAS_REQUESTS:
            return False
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = None,
             max_tokens: int = None) -> str:
        """Send chat request to local model server."""
        if not HAS_REQUESTS:
            raise LLMConnectionError("requests library not installed")
        
        temperature = temperature if temperature is not None else 0.1
        max_tokens = max_tokens if max_tokens is not None else 1024
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise LLMConnectionError(f"Cannot connect to local model at {self.base_url}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError(f"Local model request timed out after {self.timeout}s")
        except Exception as e:
            raise LLMError(f"Local model error: {e}")


# =============================================================================
# Unified LLM Client
# =============================================================================

class LLMClient:
    """
    Unified LLM client that auto-selects provider based on config.
    
    Usage:
        client = LLMClient()
        
        # Simple chat
        response = client.chat("Hello, how are you?")
        
        # With messages
        response = client.chat_messages([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ])
        
        # Check availability
        if client.is_available():
            response = client.chat("...")
    """
    
    PROVIDERS = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "azure": AzureOpenAIProvider,
        "local": LocalModelProvider
    }
    
    def __init__(self, provider: str = None):
        """
        Initialize LLM client.
        
        Args:
            provider: Provider name (ollama/openai/azure/local).
                     If None, reads from config.
        """
        self.provider_name = provider or _get_cfg("llm.provider", "ollama")
        
        if self.provider_name not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {self.provider_name}. "
                           f"Available: {list(self.PROVIDERS.keys())}")
        
        self.provider = self.PROVIDERS[self.provider_name]()
        logger.info(f"LLM Client initialized with provider: {self.provider_name}")
    
    def is_available(self) -> bool:
        """Check if LLM provider is available."""
        return self.provider.is_available()
    
    def chat(self, 
             prompt: str, 
             system_prompt: str = None,
             temperature: float = None,
             max_tokens: int = None) -> str:
        """
        Simple chat with a single prompt.
        
        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        
        Returns:
            LLM response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_messages(messages, temperature, max_tokens)
    
    def chat_messages(self,
                      messages: List[Dict[str, str]],
                      temperature: float = None,
                      max_tokens: int = None) -> str:
        """
        Chat with full message history.
        
        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            LLM response text
        """
        return self.provider.chat(messages, temperature, max_tokens)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider."""
        info = {
            "provider": self.provider_name,
            "available": self.is_available()
        }
        
        if hasattr(self.provider, 'model'):
            info["model"] = self.provider.model
        if hasattr(self.provider, 'base_url'):
            info["base_url"] = self.provider.base_url
            
        return info


# =============================================================================
# Convenience Functions
# =============================================================================

_default_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    """Get or create the default LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def quick_chat(prompt: str, system_prompt: str = None) -> str:
    """Quick one-off chat using default client."""
    client = get_llm_client()
    return client.chat(prompt, system_prompt)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== LLM Client Demo ===\n")
    
    client = LLMClient()
    info = client.get_provider_info()
    
    print(f"Provider: {info['provider']}")
    print(f"Available: {info['available']}")
    if 'model' in info:
        print(f"Model: {info['model']}")
    if 'base_url' in info:
        print(f"Base URL: {info['base_url']}")
    
    print()
    
    if client.is_available():
        print("Testing chat...")
        try:
            response = client.chat(
                "Please respond with exactly: 'Hello from LLM!'",
                system_prompt="You are a helpful assistant. Follow instructions exactly."
            )
            print(f"Response: {response}")
        except LLMError as e:
            print(f"Error: {e}")
    else:
        print(f"Provider '{info['provider']}' is not available.")
        print("Please ensure:")
        if info['provider'] == 'ollama':
            print("  - Ollama is running: ollama serve")
            print(f"  - Model is pulled: ollama pull {info.get('model', 'qwen2.5:7b')}")
        elif info['provider'] == 'openai':
            print("  - OPENAI_API_KEY environment variable is set")
        elif info['provider'] == 'azure':
            print("  - AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set")

