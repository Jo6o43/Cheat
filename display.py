import cv2


class Display:
    def __init__(self, window_name="YOLO Detection Feed", scale: float = 2.0):
        self.window_name = window_name
        self.scale = float(scale)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, frame, detections):
        for p in detections:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if p["head"]:
                hx, hy = p["head"]
                cv2.circle(frame, (hx, hy), 6, (0, 255, 0), -1)

        scaled = self.get_scaled_frame(frame)
        cv2.imshow(self.window_name, scaled)
        try:
            cv2.resizeWindow(self.window_name, scaled.shape[1], scaled.shape[0])
        except Exception:
            pass

    def get_scaled_frame(self, frame):
        if self.scale != 1.0:
            try:
                return cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            except Exception:
                return frame
        return frame

    def should_quit(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def close(self):
        cv2.destroyAllWindows()