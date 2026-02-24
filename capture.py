import cv2
import numpy as np
from mss import mss


class ColorDetector:
    def __init__(self, search_size: int = 400, lower_color: np.ndarray = None, upper_color: np.ndarray = None):
        self.sct = mss()
        self.search_size = int(search_size)
        self.monitor = self.sct.monitors[1]

        self.screen_w = self.monitor['width']
        self.screen_h = self.monitor['height']

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
        frame_suave = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(frame_suave, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_color.astype('uint8'), self.upper_color.astype('uint8'))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.medianBlur(mask, 5)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        if contours:
            # Filter out contours that look like UI elements (health bars are
            # typically wide and very short) or are too small to be an enemy.
            candidates = []
            min_area = 300  # ignore tiny blobs
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h + 1e-6)
                # Health/HP bars tend to be very wide but short (high aspect).
                if aspect > 3.0:
                    continue
                if h < 12:
                    # too short to be a character body
                    continue
                candidates.append((area, cnt))

            if candidates:
                largest = max(candidates, key=lambda t: t[0])[1]
            else:
                # fallback to the largest contour if nothing passed filters
                largest = max(contours, key=cv2.contourArea)

            # 2. Get the Bounding Box (coordinates are relative to the captured frame)
            x, y, w, h = cv2.boundingRect(largest)

            # 3. Calculate Head Position (relative to the frame)
            head_x = int(x + (w / 2))
            head_y = int(y + int(h * 0.15))

            # Use frame-relative coordinates so drawing appears on the captured image
            detections.append({
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "head": [int(head_x), int(head_y)]
            })

        return frame, detections, mask