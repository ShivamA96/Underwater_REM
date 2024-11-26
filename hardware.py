import cv2
import numpy as np
import serial
from collections import deque
import json
import os
from datetime import datetime
import statistics
import time

# Load COCO class names
COCO_CLASSES = [
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


class FishTracker:
    def __init__(self, weights_path, config_path, log_dir='fish_logs'):
        # Video capture initialization
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FPS, 60)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap = cv2.VideoCapture(0)
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = COCO_CLASSES
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        # self.ultrasonic_serial = serial.Serial(
        #     'COM3', 9600)  # Adjust port as needed
        self.previous_positions = []
        self.average_velocity = 0

        # Ultrasonic sensor initialization
        try:
            self.ultrasonic_serial = serial.Serial('COM3', 9600, timeout=1)
        except Exception as e:
            print("Error initializing ultrasonic sensor:", e)
            self.ultrasonic_serial = None

        # YOLO network loading
        self.net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1]
                              for i in self.net.getUnconnectedOutLayers()]

        # Tracking variables
        self.fish_tracking = {}  # Track individual fish
        self.all_fish_velocities = []  # Store velocities of all fish

        # Logging setup
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Timing for velocity calculation
        self.last_frame_time = time.time()

    def get_ultrasonic_distance(self):
        """Get distance from ultrasonic sensor"""
        if not self.ultrasonic_serial:
            return 1  # Default distance if sensor unavailable
        try:
            self.ultrasonic_serial.write(b'R')  # Send request
            distance = self.ultrasonic_serial.readline().decode('utf-8').strip()
            return float(distance) if distance else 1
        except Exception as e:
            print("Ultrasonic sensor error:", e)
            return 1

    def calculate_fish_velocity(self, fish_id):
        """Calculate velocity for a specific fish with more robust method"""
        fish_data = self.fish_tracking[fish_id]
        positions = fish_data['positions']

        # Require at least 2 positions and some time elapsed
        if len(positions) < 2:
            return 0

        # Calculate time elapsed
        time_elapsed = time.time() - fish_data['first_seen']
        if time_elapsed == 0:
            return 0

        # Calculate total displacement
        start_pos = positions[0]
        end_pos = positions[-1]

        # Calculate displacement in 3D space (x, y, depth)
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dz = end_pos[2] - start_pos[2]

        # Calculate Euclidean distance
        displacement = np.sqrt(dx**2 + dy**2 + dz**2)

        # Velocity = total displacement / time elapsed
        velocity = displacement / time_elapsed

        # Debug print for understanding velocity calculation
        print(f"Fish {fish_id} Velocity Calculation:")
        print(f"  Start Position: {start_pos}")
        print(f"  End Position: {end_pos}")
        print(f"  Displacement: {displacement}")
        print(f"  Time Elapsed: {time_elapsed}")
        print(f"  Calculated Velocity: {velocity} m/s")

        return velocity

    def save_velocity_logs(self):
        """Save detailed velocity logs with enhanced error handling"""
        velocity_log = {
            'timestamp': datetime.now().isoformat(),
            'individual_fish_velocities': {},
            'average_velocity': 0,
            'max_velocity': 0,
            'min_velocity': 0
        }

        # Calculate velocities for each tracked fish
        fish_velocities = []
        for fish_id in list(self.fish_tracking.keys()):
            try:
                velocity = self.calculate_fish_velocity(fish_id)

                # Only add non-zero velocities
                if velocity > 0:
                    fish_velocities.append(velocity)
                    velocity_log['individual_fish_velocities'][fish_id] = velocity
            except Exception as e:
                print(f"Error calculating velocity for fish {fish_id}: {e}")

        # Compute statistical metrics if velocities exist
        if fish_velocities:
            velocity_log['average_velocity'] = statistics.mean(fish_velocities)
            velocity_log['max_velocity'] = max(fish_velocities)
            velocity_log['min_velocity'] = min(fish_velocities)

        # Store for historical tracking
        self.all_fish_velocities.append(velocity_log)

        # Save to file
        log_path = os.path.join(
            self.log_dir, 'comprehensive_velocity_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.all_fish_velocities, f, indent=4)

        return velocity_log

    def detect_fish(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                height, width, channels = frame.shape

                # Object detection
                blob = cv2.dnn.blobFromImage(
                    frame, 0.00392, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                outs = self.net.forward(self.output_layers)

                # Information on the screen
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

                positions = []
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        label = str(self.classes[0])
                        confidence = confidences[i]
                        positions.append((x + w // 2, y + h // 2))
                        cv2.rectangle(
                            frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                    # Calculate average velocity
                    if self.previous_positions:
                        velocities = []
                        for prev_pos, curr_pos in zip(self.previous_positions, positions):
                            dx = curr_pos[0] - prev_pos[0]
                            dy = curr_pos[1] - prev_pos[1]
                            velocity = np.sqrt(dx**2 + dy**2)
                            velocities.append(velocity)
                        self.average_velocity = sum(
                            velocities) / len(velocities)
                    self.previous_positions = positions
                else:
                    self.average_velocity = 0
                    self.previous_positions = []

                # Display average velocity
                cv2.putText(frame, f"Average Velocity: {self.average_velocity:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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


# Run the fish tracker
if __name__ == "__main__":
    tracker = FishTracker('./yolov3.weights', './yolov3.cfg')
    tracker.detect_fish()
