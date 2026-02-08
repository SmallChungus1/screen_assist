import mss
from PIL import Image
from typing import Optional
import sys

def capture_fullscreen(monitor: Optional[int] = None) -> Image.Image:
    with mss.mss() as sct:
        if monitor is None:
            monitor_idx = 1
        else:
            monitor_idx = monitor
            
        try:
            sct_img = sct.grab(sct.monitors[monitor_idx])
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Check for permission issues (all black or empty)
            # This is a basic heuristic.
            if not img.getbbox():
                 print("Warning: Screenshot appears blank. Check usage permissions.")
                 print("Enable Screen & System Audio Recording permission for this app in System Settings -> Privacy & Security.")
            
            return img
        except IndexError:
             print(f"Monitor {monitor_idx} not found, defaulting to 1")
             sct_img = sct.grab(sct.monitors[1])
             img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
             return img
        except Exception as e:
            print(f"Capture failed: {e}")
            print("Enable Screen & System Audio Recording permission for this app in System Settings -> Privacy & Security.")
            raise e
