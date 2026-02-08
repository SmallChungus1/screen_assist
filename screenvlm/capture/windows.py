import mss
from PIL import Image
from typing import Optional

def capture_fullscreen(monitor: Optional[int] = None) -> Image.Image:
    with mss.mss() as sct:
        if monitor is None:
            monitor_idx = -1  # All monitors
            # But spec says "primary", mss uses 1 for primary? 
            # Actually mss: 1 is primary. -1 is all.
            # Let's default to primary content monitor if not specified? 
            # Or usually users want the screen they are looking at.
            # To be safe and simple, let's use monitor 1 (primary) as default if None.
            monitor_idx = 1 
        else:
            monitor_idx = monitor
            
        # mss monitors list: 0 is all, 1 is 1st, 2 is 2nd...
        # If user asks for 1, we give 1.
        
        try:
            sct_img = sct.grab(sct.monitors[monitor_idx])
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            return img
        except IndexError:
             # Fallback to 1 if index out of bounds
             print(f"Monitor {monitor_idx} not found, defaulting to 1")
             sct_img = sct.grab(sct.monitors[1])
             img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
             return img
