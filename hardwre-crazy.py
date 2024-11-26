# hardware.py

import cv2
import numpy as np
import serial
from collections import defaultdict


class FishTracker:
    def __init__(self, weights_path, config_path):
        # Load YOLO
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush", "fish"
        ]

        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1]
                              for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.frame_count = 0
        # Ultrasonic sensor serial port
        self.ultrasonic_serial = None
        # Data for calculations
        self.fish_positions = defaultdict(list)
        self.encounters = 0
        self.prev_frame_empty = True
        self.next_id = 0

    def detect_fish(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame_count += 1
                height, width, channels = frame.shape
                # Detecting objects
                blob = cv2.dnn.blobFromImage(
                    frame, 0.00392, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                outs = self.net.forward(self.output_layers)
                # Showing informations on the screen
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5 and self.classes[class_id] == 'cell phone':
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                fish_in_frame = False
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        label = str(self.classes[class_ids[i]])
                        color = self.colors[class_ids[i]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y + 30),
                                    cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
                        # Track fish positions
                        fish_in_frame = True
                        cx, cy = x + w // 2, y + h // 2
                        fish_id = self.assign_id(cx, cy)
                        self.fish_positions[fish_id].append(
                            (self.frame_count, cx, cy))
                # Encounter detection
                if fish_in_frame and self.prev_frame_empty:
                    self.encounters += 1
                self.prev_frame_empty = not fish_in_frame
                cv2.imshow("Fish Detection", frame)
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            if self.ultrasonic_serial:
                self.ultrasonic_serial.close()
            self.calculate_average_velocities()

    def assign_id(self, cx, cy):
        # Simple ID assignment based on proximity
        for fish_id, positions in self.fish_positions.items():
            last_frame, last_cx, last_cy = positions[-1]
            if abs(cx - last_cx) < 50 and abs(cy - last_cy) < 50:
                return fish_id
        self.next_id += 1
        return self.next_id

    def calculate_average_velocities(self):
        total_velocity = 0
        total_count = 0
        for positions in self.fish_positions.values():
            for i in range(1, len(positions)):
                frame_diff = positions[i][0] - positions[i-1][0]
                dist = np.hypot(positions[i][1] - positions[i-1][1],
                                positions[i][2] - positions[i-1][2])
                velocity = dist / frame_diff if frame_diff > 0 else 0
                total_velocity += velocity
                total_count += 1
        if total_count > 0:
            average_velocity = total_velocity / total_count
            print(
                f"Average velocity of all fishes: {average_velocity:.2f} pixels/frame")
        else:
            print("No velocity data available.")
        print(f"Total encounters: {self.encounters}")


# Run the fish tracker
if __name__ == "__main__":
    tracker = FishTracker('./yolov3.weights', './yolov3.cfg')
    tracker.detect_fish()
