"""
Configuration management utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """
    Manage application configuration from multiple sources.
    
    Examples:
        >>> config = ConfigManager("config.yml")
        >>> db_settings = config.get("database")
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary of configuration values
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        if self.config_path.suffix in ['.yaml', '.yml']:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        elif self.config_path.suffix == '.json':
            with open(self.config_path) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {self.config_path.suffix}")