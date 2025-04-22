"""
Advanced file system operations and management.
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Generator

class FileManager:
    """
    Manage file operations with error handling and logging.
    
    Examples:
        >>> fm = FileManager(base_dir="project/data")
        >>> fm.create_directory("processed")
        >>> fm.copy_files("*.csv", "processed/")
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_directory(self, dir_name: str) -> Path:
        """
        Create a new directory.
        
        Args:
            dir_name: Name of directory to create
            
        Returns:
            Path object of created directory
        """
        new_dir = self.base_dir / dir_name
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir