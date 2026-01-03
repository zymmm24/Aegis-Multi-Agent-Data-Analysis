"""
Configuration Loader for Aegis Multi-Agent Data Analysis System

Provides unified configuration management with:
- YAML config file loading
- Environment variable substitution
- Default value fallbacks
- Type-safe access to configuration values
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

# Try to import yaml, fallback to json if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    import json
    HAS_YAML = False
    

logger = logging.getLogger("ConfigLoader")


class ConfigurationError(Exception):
    """Raised when configuration loading or access fails."""
    pass


def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute ${VAR} patterns with environment variables.
    
    Examples:
        "${OPENAI_API_KEY}" -> actual value from os.environ
        "${VAR:-default}" -> value or "default" if not set
    """
    if isinstance(value, str):
        # Pattern: ${VAR} or ${VAR:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            # Return original if not found and no default
            return match.group(0)
        
        return re.sub(pattern, replacer, value)
    
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    
    return value


class Config:
    """
    Configuration manager with dot-notation access.
    
    Usage:
        config = Config.load()
        
        # Access nested values
        model = config.get("llm.ollama.model")
        threshold = config.get("l1_etl.confidence_thresholds.high", default=0.75)
        
        # Or use attribute access
        config.llm.provider
    """
    
    _instance: Optional['Config'] = None
    _config_data: Dict[str, Any] = {}
    
    def __init__(self, config_data: Dict[str, Any]):
        self._config_data = config_data
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file. If None, searches for:
                1. ./config.yaml
                2. ./config.yml
                3. ./config.json
        
        Returns:
            Config instance
        """
        if cls._instance is not None:
            return cls._instance
        
        # Find config file
        if config_path is None:
            search_paths = [
                Path("config.yaml"),
                Path("config.yml"),
                Path("config.json"),
                Path(__file__).parent / "config.yaml",
                Path(__file__).parent / "config.yml",
            ]
            for path in search_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path is None or not Path(config_path).exists():
            logger.warning("No config file found, using defaults")
            cls._instance = cls(_get_default_config())
            return cls._instance
        
        # Load config file
        config_path = Path(config_path)
        logger.info(f"Loading configuration from: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix in ('.yaml', '.yml'):
                    if HAS_YAML:
                        data = yaml.safe_load(f)
                    else:
                        raise ConfigurationError("PyYAML not installed. Run: pip install pyyaml")
                else:
                    data = json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file: {e}")
        
        # Substitute environment variables
        data = _substitute_env_vars(data)
        
        cls._instance = cls(data)
        return cls._instance
    
    @classmethod
    def reload(cls, config_path: Optional[str] = None) -> 'Config':
        """Force reload configuration."""
        cls._instance = None
        return cls.load(config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key path (e.g., "llm.ollama.model")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section as dict."""
        return self.get(section, {})
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to top-level config sections."""
        if name.startswith('_'):
            raise AttributeError(name)
        
        value = self._config_data.get(name)
        if isinstance(value, dict):
            return _ConfigSection(value)
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config_data.copy()


class _ConfigSection:
    """Helper class for attribute-style access to nested config."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        
        value = self._data.get(name)
        if isinstance(value, dict):
            return _ConfigSection(value)
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()


def _get_default_config() -> Dict[str, Any]:
    """Return default configuration when no config file is found."""
    return {
        "l1_etl": {
            "chunk_rows": 10000,
            "sniff_bytes": 65536,
            "max_header_scan": 8,
            "header_lookahead": 20,
            "score_gap_threshold": 0.40,
            "confidence_thresholds": {
                "high": 0.75,
                "medium": 0.50,
                "low": 0.30
            },
            "output_dir": "ir_output"
        },
        "llm": {
            "provider": "ollama",
            "auto_review_on_low_confidence": True,
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "qwen2.5:7b",
                "timeout": 60
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(levelname)s:%(name)s:%(message)s"
        }
    }


# Convenience function for quick access
def get_config() -> Config:
    """Get or load the global configuration instance."""
    return Config.load()


# -----------------------------------------------------------------------------
# Usage Examples
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Demo usage
    config = Config.load()
    
    print("=== Configuration Demo ===\n")
    
    # Dot-notation access
    print(f"LLM Provider: {config.get('llm.provider')}")
    print(f"Ollama Model: {config.get('llm.ollama.model')}")
    print(f"High Confidence Threshold: {config.get('l1_etl.confidence_thresholds.high')}")
    
    # Attribute access
    print(f"\nUsing attribute access:")
    print(f"  llm.provider = {config.llm.provider}")
    print(f"  l1_etl.chunk_rows = {config.l1_etl.chunk_rows}")
    
    # Get with default
    print(f"\nWith default: {config.get('nonexistent.key', 'default_value')}")
    
    # Get section
    print(f"\nL1 ETL Section: {config.get_section('l1_etl')}")

