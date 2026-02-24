import cv2
import numpy as np
from mss import mss


class ColorDetector:
    def __init__(self, search_size: int = 200, lower_color: np.ndarray = None, upper_color: np.ndarray = None, exclude_yellow: bool = True):
        self.sct = mss()
        self.max_aspect = 2.5  # Default aspect ratio threshold (HP bars are wider)
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
            candidates = []
            min_area = 300
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]

            def _is_hp_bar(cnt, area, x, y, w, h):
                """Return True if this contour looks like an HP/status bar."""
                aspect = w / float(h + 1e-6)

                # 1) Very high aspect ratio -> almost certainly a bar
                if aspect > self.max_aspect:
                    return True

                # 2) Absolute height too small to be an enemy body
                if h < 12:
                    return True

                # 3) Height is unrealistically small compared to width
                #    (catches bars whose aspect just sneaks under max_aspect)
                if h < max(8, int(w * 0.25)):
                    return True

                # 4) Wide + very short relative to search area -> UI bar
                if w > int(self.search_size * 0.6) and h < max(6, int(self.search_size * 0.08)):
                    return True

                # 5) Solidity check adapted for SEGMENTED bars.
                #    A segmented bar (many rounded blocks with gaps) has solidity
                #    in the 0.50–0.82 range — LOWER than a solid body but still
                #    wide.  A body silhouette tends to be taller and more varied.
                bbox_area = float(w * h + 1e-6)
                solidity  = area / bbox_area
                # Solid-block bar (gaps filled by morphology) OR segmented bar
                if aspect > 2.0 and 0.45 < solidity < 0.95:
                    return True

                # 6) Convexity-defect count: the rounded segments create many
                #    periodic concavities (notches) along the bar's top/bottom.
                #    Count significant defects — bars have >= 4, bodies far fewer
                #    per unit aspect.
                try:
                    hull_idx = cv2.convexHull(cnt, returnPoints=False)
                    if hull_idx is not None and len(hull_idx) >= 3:
                        defects = cv2.convexityDefects(cnt, hull_idx)
                        if defects is not None:
                            # Count defects whose depth > 2 pixels
                            sig_defects = int(np.sum(defects[:, 0, 3] / 256.0 > 2.0))
                            # Many small periodic defects + wide/short = segmented bar
                            if sig_defects >= 4 and aspect > 1.5:
                                return True
                except cv2.error:
                    pass

                # 7) Both V-channel AND S-channel variance must be low for a bar.
                #    Bars are uniformly coloured; enemy textures vary in both.
                mask_c = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(mask_c, [cnt], -1, 255, -1)
                _, sv = cv2.meanStdDev(v_channel, mask=mask_c)
                _, ss = cv2.meanStdDev(s_channel, mask=mask_c)
                std_v = float(sv[0, 0]) if sv is not None else 0.0
                std_s = float(ss[0, 0]) if ss is not None else 0.0
                # Low variance in BOTH channels -> flat coloured region (bar)
                if std_v < 9.0 and std_s < 14.0:
                    return True
                # Still reject if brightness alone is very flat
                if std_v < 5.0:
                    return True

                return False

            # ── Build a rect lookup for every valid contour ─────────────────
            all_rects = []
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a < min_area:
                    continue
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                all_rects.append((cnt, a, rx, ry, rw, rh))

            # ── Bar-above-body suppression ───────────────────────────────────
            # If a wide/short contour's bottom edge sits just above the top of a
            # taller, narrower contour it is the HP bar floating above the body.
            def _is_bar_above_body(x, y, w, h):
                if w < h * 1.5:           # not wide enough to be a bar candidate
                    return False
                bar_bottom = y + h
                bar_mid    = x + w // 2
                gap_tol    = max(12, int(self.search_size * 0.06))
                for _, _, bx, by, bw, bh in all_rects:
                    if bh <= h:            # body must be taller than the bar
                        continue
                    if not (bar_bottom <= by <= bar_bottom + gap_tol):
                        continue
                    if bx <= bar_mid <= bx + bw:   # horizontally aligned
                        return True
                return False

            for cnt, area, x, y, w, h in all_rects:
                if _is_hp_bar(cnt, area, x, y, w, h):
                    continue
                if _is_bar_above_body(x, y, w, h):
                    continue
                candidates.append((area, cnt))

            if candidates:
                largest = max(candidates, key=lambda t: t[0])[1]
            else:
                # Fallback: use largest contour that at least isn't paper-thin
                fallback = [
                    cnt for cnt, a, rx, ry, rw, rh in all_rects
                    if rh >= 8 and (rw / float(rh + 1e-6)) <= self.max_aspect
                ]
                largest = (
                    max(fallback, key=cv2.contourArea) if fallback
                    else max(contours, key=cv2.contourArea)
                )

            x, y, w, h = cv2.boundingRect(largest)
            head_x = int(x + w / 2)
            head_y = int(y + h * 0.15)

            detections.append({
                "box":  [int(x), int(y), int(x + w), int(y + h)],
                "head": [int(head_x), int(head_y)]
            })

        return frame, detections, mask