import cv2
import numpy as np
from mss import mss


# ── Helpers adapted from CVM-colorBot/src/utils/detection.py ─────────────────

def _has_vertical_color_line(mask, cx, y_top, y_bot, min_hits: int = 3):
    """
    Return True if the vertical column at x=cx between y_top and y_bot
    contains at least `min_hits` non-zero (colour-detected) pixels.

    Enemy bodies span many rows of orange pixels → pass.
    HP bars are only a few pixels tall → fail (too few rows).
    """
    cx     = int(np.clip(cx,    0, mask.shape[1] - 1))
    y_top  = int(np.clip(y_top, 0, mask.shape[0]))
    y_bot  = int(np.clip(y_bot, 0, mask.shape[0]))
    if y_bot <= y_top:
        return False
    col = mask[y_top:y_bot, cx]
    return int(np.count_nonzero(col)) >= min_hits


def _bbox_area(x, y, w, h):
    return max(0, int(w)) * max(0, int(h))


def _boxes_overlap(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 < x2 + w2 and x1 + w1 > x2) and (y1 < y2 + h2 and y1 + h1 > y2)


def _overlap_len(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _should_merge(r1, r2, dist_thr):
    """Merge two rects if they overlap or one is a small fragment close to the other."""
    if _boxes_overlap(r1, r2):
        return True
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    x1r, y1r = x1 + w1, y1 + h1
    x2r, y2r = x2 + w2, y2 + h2

    a1 = max(1, _bbox_area(*r1))
    a2 = max(1, _bbox_area(*r2))
    if min(a1, a2) / max(a1, a2) > 0.80:   # similarly sized → different entities
        return False

    h_gap     = max(0, max(x1, x2) - min(x1r, x2r))
    v_gap     = max(0, max(y1, y2) - min(y1r, y2r))
    h_overlap = _overlap_len(x1, x1r, x2, x2r)
    v_overlap = _overlap_len(y1, y1r, y2, y2r)

    close_h = h_gap <= dist_thr and v_overlap >= 0.60 * min(h1, h2)
    close_v = v_gap <= dist_thr and h_overlap >= 0.60 * min(w1, w2)
    return close_h or close_v


def _merge_rects(rects, dist_thr=12):
    """Merge fragmented contour boxes into unified player bounding boxes."""
    used   = [False] * len(rects)
    merged = []

    for i, r1 in enumerate(rects):
        if used[i]:
            continue
        nx1, ny1 = r1[0], r1[1]
        nx2, ny2 = r1[0] + r1[2], r1[1] + r1[3]

        changed = True
        while changed:
            changed = False
            cur = (nx1, ny1, nx2 - nx1, ny2 - ny1)
            for j, r2 in enumerate(rects):
                if i == j or used[j]:
                    continue
                if _should_merge(cur, r2, dist_thr):
                    nx1, ny1 = min(nx1, r2[0]), min(ny1, r2[1])
                    nx2, ny2 = max(nx2, r2[0] + r2[2]), max(ny2, r2[1] + r2[3])
                    used[j]  = True
                    changed  = True
        used[i] = True
        merged.append((nx1, ny1, max(1, nx2 - nx1), max(1, ny2 - ny1)))

    return sorted(merged, key=lambda r: _bbox_area(*r), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────

class ColorDetector:
    def __init__(self,
                 search_size: int = 200,
                 lower_color: np.ndarray = None,
                 upper_color: np.ndarray = None,
                 exclude_yellow: bool = True):

        self.sct          = mss()
        self.search_size  = int(search_size)
        self.monitor      = self.sct.monitors[1]
        self.screen_w     = self.monitor['width']
        self.screen_h     = self.monitor['height']

        # Aspect-ratio cap: HP bars are very wide/short.
        # Keep this relatively generous; the vertical-line check does the heavy lifting.
        self.max_aspect   = 4.0

        # Minimum number of orange pixels along the centre vertical column.
        # An HP bar (height ≈ 4–12 px) will rarely reach this.
        # An enemy body (height ≥ 30 px) easily exceeds it.
        self.min_vert_hits = 6

        # Default: orange-biased HSV range
        self.lower_color = (np.array([6, 120, 120])   if lower_color is None
                            else np.array(lower_color, dtype=np.int32))
        self.upper_color = (np.array([18, 255, 255])  if upper_color is None
                            else np.array(upper_color, dtype=np.int32))

        self.exclude_yellow = bool(exclude_yellow)

    # ── Public setters ────────────────────────────────────────────────────────

    def set_thresholds(self, lower, upper):
        self.lower_color = np.array(lower, dtype=np.int32)
        self.upper_color = np.array(upper, dtype=np.int32)

    def set_search_size(self, size: int):
        self.search_size = int(size)

    # ── Main detection method ─────────────────────────────────────────────────

    def get_data(self):
        # 1. Screen capture centred on crosshair
        region = {
            "top":    int(self.screen_h // 2 - self.search_size // 2),
            "left":   int(self.screen_w // 2 - self.search_size // 2),
            "width":  self.search_size,
            "height": self.search_size,
        }
        img        = np.array(self.sct.grab(region))
        frame      = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv        = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # 2. Build HSV mask
        lower = np.array(self.lower_color, dtype=np.int32)
        upper = np.array(self.upper_color, dtype=np.int32)
        lower[0]  = int(np.clip(lower[0], 0, 179))
        upper[0]  = int(np.clip(upper[0], 0, 179))
        lower[1:] = np.clip(lower[1:], 0, 255)
        upper[1:] = np.clip(upper[1:], 0, 255)

        if lower[0] <= upper[0]:
            mask = cv2.inRange(hsv, lower.astype('uint8'), upper.astype('uint8'))
        else:
            # Hue wrap-around (e.g. red spans H≈170–10)
            m1   = cv2.inRange(hsv,
                               np.array([lower[0], lower[1], lower[2]], dtype=np.uint8),
                               np.array([179,       upper[1], upper[2]], dtype=np.uint8))
            m2   = cv2.inRange(hsv,
                               np.array([0,         lower[1], lower[2]], dtype=np.uint8),
                               np.array([upper[0],  upper[1], upper[2]], dtype=np.uint8))
            mask = cv2.bitwise_or(m1, m2)

        raw_mask = mask.copy()                           # keep raw for fallback

        # 3. Subtract yellow floor markings / UI
        if self.exclude_yellow:
            yellow_mask = cv2.inRange(hsv,
                                      np.array([18, 80, 80],   dtype=np.uint8),
                                      np.array([35, 255, 255], dtype=np.uint8))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(yellow_mask))

        # 4. Morphology: close gaps, then gentle open to remove noise.
        #    A wider horizontal kernel helps merge the segmented HP-bar blocks
        #    into ONE large blob — which is then caught by the vertical-line check.
        close_k = np.ones((5, 15), np.uint8)             # wide horizontal close
        open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k)
        mask    = cv2.medianBlur(mask, 3)

        # Recover if morphology killed everything
        if cv2.countNonZero(mask) == 0 and cv2.countNonZero(raw_mask) > 0:
            mask = raw_mask

        # 5. Find contours
        cnts      = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours  = cnts[0] if len(cnts) == 2 else cnts[1]

        # 6. Filter contours
        img_h, img_w = frame.shape[:2]
        frame_area   = max(1, img_h * img_w)

        min_contour_area = max(50,  int(frame_area * 0.00008))
        min_bbox_area    = max(100, int(frame_area * 0.00025))
        min_fill_ratio   = 0.04      # contour must fill ≥4 % of its bounding box
        min_h_px         = 10        # must be at least 10 px tall to be a body
        border_margin    = 1

        valid_rects = []

        for cnt in contours:
            if len(cnt) < 5:
                continue

            contour_area = float(cv2.contourArea(cnt))
            if contour_area < min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            bb_area     = int(w) * int(h)

            if bb_area < min_bbox_area:
                continue
            if h < min_h_px:
                continue

            # Fill ratio: sparse contours are noise
            if contour_area / max(float(bb_area), 1.0) < min_fill_ratio:
                continue

            # Aspect ratio: very wide/short → HP bar
            aspect = float(w) / max(float(h), 1.0)
            if aspect > self.max_aspect:
                continue

            # Edge-touching check: tiny border blobs are usually artefacts
            touches = (x <= border_margin or y <= border_margin or
                       (x + w) >= (img_w - border_margin) or
                       (y + h) >= (img_h - border_margin))
            if touches and bb_area < max(200, int(frame_area * 0.0006)):
                continue

            # ── KEY FILTER (from CVM-colorBot) ───────────────────────────────
            # Enemy bodies have a tall vertical column of orange pixels.
            # HP bars are only a few pixels tall and CANNOT pass this test.
            cx = x + w // 2
            if not _has_vertical_color_line(mask, cx, y, y + h,
                                            min_hits=self.min_vert_hits):
                continue
            # ─────────────────────────────────────────────────────────────────

            valid_rects.append((x, y, w, h))

        # 7. Fallback: if strict pass found nothing, try with only the
        #    vertical-line check (no size or aspect guards)
        if not valid_rects and cv2.countNonZero(raw_mask) > 0:
            raw_cnts = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            raw_contours = raw_cnts[0] if len(raw_cnts) == 2 else raw_cnts[1]
            loose_min    = max(20, int(min_bbox_area * 0.5))
            for cnt in raw_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if int(w) * int(h) < loose_min:
                    continue
                cx = x + w // 2
                if _has_vertical_color_line(mask, cx, y, y + h,
                                            min_hits=max(2, self.min_vert_hits // 2)):
                    valid_rects.append((x, y, w, h))

        # 8. Merge fragmented blobs (body split into upper/lower halves, etc.)
        if valid_rects:
            merged = _merge_rects(valid_rects, dist_thr=14)
        else:
            return frame, [], mask

        # 9. Pick the largest merged rect as the primary target
        x, y, w, h = merged[0]
        head_x      = int(x + w / 2)
        head_y      = int(y + h * 0.15)

        detections = [{
            "box":  [int(x), int(y), int(x + w), int(y + h)],
            "head": [int(head_x), int(head_y)],
        }]

        return frame, detections, mask