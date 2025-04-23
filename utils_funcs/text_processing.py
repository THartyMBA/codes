"""
Text processing and analysis utilities.
"""

import re
import string
from collections import Counter
from typing import List, Dict

class TextProcessor:
    """
    Process and analyze text data.
    
    Examples:
        >>> tp = TextProcessor()
        >>> cleaned = tp.clean_text("Hello, World!")
        >>> tokens = tp.tokenize(cleaned)
    """
    
    def __init__(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but'])
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing punctuation and normalizing.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()