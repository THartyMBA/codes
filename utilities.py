"""
Core Utility Functions
=====================

This module provides fundamental utility functions for common programming tasks.
Each function is designed to be standalone and reusable across different projects.

Module Contents:
    - File Operations
    - URL Handling
    - Timestamp Management
    - Logging Setup
    - Mathematical Operations
    - Error Handling
"""

def read_text_file(filepath: str) -> str:
    """
    Reads and returns the contents of a text file.
    
    Parameters:
        filepath (str): Path to the text file to be read
        
    Returns:
        str: Contents of the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the program lacks read permissions
        
    Example:
        >>> content = read_text_file("example.txt")
        >>> print(content)
        'Hello, World!'
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_text_file(filepath: str, data: str) -> None:
    """
    Writes text data to a file, creating the file if it doesn't exist.
    
    Parameters:
        filepath (str): Path where the file should be written
        data (str): Content to write to the file
        
    Returns:
        None
        
    Raises:
        PermissionError: If the program lacks write permissions
        
    Example:
        >>> write_text_file("output.txt", "Hello, World!")
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data)

def fetch_url(url: str, timeout: int = 15) -> str:
    """
    Retrieves content from a specified URL.
    
    Parameters:
        url (str): The URL to fetch content from
        timeout (int): Maximum time to wait for response in seconds
        
    Returns:
        str: Content of the URL response
        
    Raises:
        requests.exceptions.RequestException: For various network-related errors
        
    Example:
        >>> content = fetch_url("https://api.example.com/data")
        >>> print(content)
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text