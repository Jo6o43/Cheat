import cv2

class Display:
    def __init__(self, window_name="YOLO Detection Feed"):
        self.window_name = window_name
        
    def show(self, frame, detections):
        for p in detections:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if p["head"]:
                hx, hy = p["head"]
                cv2.circle(frame, (hx, hy), 6, (0, 255, 0), -1)

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        cv2.imshow(self.window_name, small_frame)

    def should_quit(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def close(self):
        cv2.destroyAllWindows()