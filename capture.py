import cv2
import numpy as np
from mss import mss


class ColorDetector:
    def __init__(self, search_size: int = 200, lower_color: np.ndarray = None, upper_color: np.ndarray = None, exclude_yellow: bool = True):
        self.sct = mss()
        self.max_aspect = 2.8  # Default aspect ratio threshold
        self.search_size = int(search_size)
        self.monitor = self.sct.monitors[1]

        self.screen_w = self.monitor['width']
        self.screen_h = self.monitor['height']

        # Default to an orange-biased range (helps prefer orange over yellow)
        self.lower_color = np.array([6, 120, 120]) if lower_color is None else np.array(lower_color, dtype=np.int32)
        self.upper_color = np.array([18, 255, 255]) if upper_color is None else np.array(upper_color, dtype=np.int32)

        # When True, pixels that strongly match a generic yellow range will be
        # removed from the main mask to avoid detecting yellow UI elements.
        self.exclude_yellow = bool(exclude_yellow)

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

        # Clamp HSV to valid ranges and handle H wrap-around (useful for red)
        lower = np.array(self.lower_color, dtype=np.int32)
        upper = np.array(self.upper_color, dtype=np.int32)
        lower[0] = int(np.clip(lower[0], 0, 179))
        upper[0] = int(np.clip(upper[0], 0, 179))
        lower[1:] = np.clip(lower[1:], 0, 255)
        upper[1:] = np.clip(upper[1:], 0, 255)

        if lower[0] <= upper[0]:
            mask = cv2.inRange(hsv, lower.astype('uint8'), upper.astype('uint8'))
        else:
            # Hue wrap-around: combine two ranges (e.g. H in [170,179] or [0,10])
            low1 = np.array([lower[0], lower[1], lower[2]], dtype=np.uint8)
            high1 = np.array([179, upper[1], upper[2]], dtype=np.uint8)
            low2 = np.array([0, lower[1], lower[2]], dtype=np.uint8)
            high2 = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)
            m1 = cv2.inRange(hsv, low1, high1)
            m2 = cv2.inRange(hsv, low2, high2)
            mask = cv2.bitwise_or(m1, m2)

        # Optionally remove generic yellow pixels to avoid false positives
        if self.exclude_yellow:
            # Generic yellow range (tunable)
            yellow_low = np.array([18, 80, 80], dtype=np.uint8)
            yellow_high = np.array([35, 255, 255], dtype=np.uint8)
            yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)
            # Subtract yellow-like pixels from the main mask
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(yellow_mask))

        # Use elliptical kernels to better preserve rounded shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.medianBlur(mask, 5)

        # Robust findContours unpacking across OpenCV versions
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        
        detections = []
        if contours:
            # Filter out contours that look like UI elements (health bars are
            # typically wide and very short) or are too small to be an enemy.
            candidates = []
            min_area = 300  # ignore tiny blobs
            v_channel = hsv[:, :, 2]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h + 1e-6)
                # Health/HP bars tend to be very wide but short (high aspect).
                if aspect > self.max_aspect:
                    continue
                if h < 12:
                    # too short to be a character body
                    continue

                # Additional heuristics to reject flat, rectangular UI bars:
                # 1) If the contour spans most of the search width but is very
                #    short in height, it's likely a UI element.
                if w > int(self.search_size * 0.6) and h < max(6, int(self.search_size * 0.08)):
                    continue

                # 2) Compute the V-channel standard deviation inside the contour.
                #    Health bars are usually flat/uniform brightness (low stddev).
                mask_contour = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(mask_contour, [cnt], -1, 255, -1)
                _, stddev = cv2.meanStdDev(v_channel, mask=mask_contour)
                std_v = float(stddev[0,0]) if stddev is not None else 0.0
                if std_v < 6.0:
                    # low brightness variance -> likely a flat UI bar
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