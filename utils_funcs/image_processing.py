"""
Image processing and manipulation utilities.
"""

from PIL import Image
import numpy as np
from typing import Tuple

class ImageProcessor:
    """
    Process and transform images.
    
    Examples:
        >>> ip = ImageProcessor()
        >>> resized_img = ip.resize_image("photo.jpg", (800, 600))
    """
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif'}
    
    def resize_image(self, 
                    image_path: str, 
                    size: Tuple[int, int],
                    save_path: str = None) -> Image.Image:
        """
        Resize image to specified dimensions.
        
        Args:
            image_path: Path to source image
            size: Tuple of (width, height)
            save_path: Optional path to save resized image
            
        Returns:
            Resized PIL Image object
        """
        img = Image.open(image_path)
        resized = img.resize(size, Image.LANCZOS)
        if save_path:
            resized.save(save_path)
        return resized