import sys
import platform

if platform.system() == "Windows":
    from .windows import capture_fullscreen
elif platform.system() == "Darwin":
    from .macos import capture_fullscreen
else:
    # Fallback or Linux support
    from .base import capture_fullscreen as base_capture
    def capture_fullscreen(monitor=None):
        print(f"Warning: Platform {platform.system()} not explicitly supported. Trying generic.")
        # Try windows/mss logic as generic
        try:
            import mss
            from PIL import Image
            with mss.mss() as sct:
                mon = monitor if monitor else 1
                sct_img = sct.grab(sct.monitors[mon])
                return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except Exception as e:
             raise NotImplementedError(f"Capture not implemented for {platform.system()}: {e}")

__all__ = ["capture_fullscreen"]
