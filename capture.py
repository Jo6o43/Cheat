import numpy as np
from mss import mss
from ultralytics import YOLO
import cv2

class Capture:
    def __init__(self, model_path='yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        self.sct = mss()
        self.monitor = self.sct.monitors[1]

    def get_data(self):
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = self.model(frame, verbose=False, conf=0.5)[0]
        
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                person = {"box": [x1, y1, x2, y2], "head": None}

                center_x = int((x1 + x2) / 2)
                
                if results.keypoints is not None:
                    kpts = results.keypoints.xy[i].cpu().numpy()
                    if len(kpts) > 0 and kpts[0][1] > 0: 
                        head_y = int(kpts[0][1])
                        person["head"] = [center_x, head_y]
                
                detections.append(person)
                
        return frame, detections