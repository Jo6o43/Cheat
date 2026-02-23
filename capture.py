import cv2
import numpy as np
from mss import mss


class ColorDetector:
    def __init__(self, search_size: int = 400, lower_color: np.ndarray = None, upper_color: np.ndarray = None):
        self.sct = mss()
        # Field of view size (square region centered on screen)
        self.search_size = int(search_size)
        self.monitor = self.sct.monitors[1]

        # Center coordinates
        self.screen_w = self.monitor['width']
        self.screen_h = self.monitor['height']

        # Default HSV thresholds (tweakable)
        self.lower_color = np.array([5, 150, 150]) if lower_color is None else np.array(lower_color, dtype=np.int32)
        self.upper_color = np.array([22, 255, 255]) if upper_color is None else np.array(upper_color, dtype=np.int32)

    def set_thresholds(self, lower, upper):
        """Set HSV lower/upper thresholds. Accepts sequences or numpy arrays."""
        self.lower_color = np.array(lower, dtype=np.int32)
        self.upper_color = np.array(upper, dtype=np.int32)

    def set_search_size(self, size: int):
        self.search_size = int(size)

    def get_data(self):
        # 1. Capture a specific region around the center
        region = {
            "top": int(self.screen_h // 2 - self.search_size // 2),
            "left": int(self.screen_w // 2 - self.search_size // 2),
            "width": self.search_size,
            "height": self.search_size
        }
        
        img = np.array(self.sct.grab(region))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Create a mask to find the color
        mask = cv2.inRange(hsv, self.lower_color.astype('uint8'), self.upper_color.astype('uint8'))
        
        # 3. Find the "Center of Mass" of the color pixels
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        if contours:
            # 1. Find the biggest blob (the enemy)
            largest = max(contours, key=cv2.contourArea)
            
            # 2. Get the Bounding Box
            # x,y is top-left corner; w,h is width and height
            x, y, w, h = cv2.boundingRect(largest)
            
            # 3. Calculate Head Position
            # Center of the blob horizontally (X)
            head_x = x + (w // 2)
            
            # Vertical Offset (Y): 
            # In most games, the head is the top ~15% of the body.
            # We take the very top 'y' and add a tiny bit of height.
            head_y = y + int(h * 0.15) 
            
            # Convert to Absolute Screen Coordinates
            abs_x = head_x + region["left"]
            abs_y = head_y + region["top"]
            
            detections.append({
                "box": [abs_x - (w//2), abs_y, abs_x + (w//2), abs_y + h],
                "head": [abs_x, abs_y]
            })

        return frame, detections, mask