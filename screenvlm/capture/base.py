from typing import Optional
from PIL import Image

def capture_fullscreen(monitor: Optional[int] = None) -> Image.Image:
    """
    Capture the full screen.
    
    Args:
        monitor: Monitor index (1-based usually). If None, capture primary.
        
    Returns:
        PIL.Image of the screenshot.
    """
    raise NotImplementedError("Platform specific capture not implemented")
