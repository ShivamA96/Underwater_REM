import cv2
import numpy as np
import serial
from collections import deque

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
    "toothbrush", "fish"  # Note: 'fish' is the last class in COCO dataset
]


class FishTracker:
    def __init__(self, weights_path, config_path):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Frame rate: {fps}")

        # Initialize ultrasonic sensor
        try:
            self.ultrasonic_serial = serial.Serial('COM3', 9600, timeout=1)
        except Exception as e:
            print("Error initializing ultrasonic sensor:", e)
            self.ultrasonic_serial = None

        # Load YOLO network
        self.net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1]
                              for i in self.net.getUnconnectedOutLayers()]

        # Initialize fish tracking
        self.fish_positions = deque(maxlen=10)

    def get_ultrasonic_distance(self):
        """Get distance from ultrasonic sensor"""
        if not self.ultrasonic_serial:
            return 1  # Return default distance if sensor unavailable
        try:
            self.ultrasonic_serial.write(b'R')  # Send request
            distance = self.ultrasonic_serial.readline().decode('utf-8').strip()
            return float(distance) if distance else 1
        except Exception as e:
            print("Ultrasonic sensor error:", e)
            return 1

    def calculate_velocity(self, positions):
        """Calculate velocity of fish movement"""
        if len(positions) < 2:
            return 0  # Not enough data

        # Calculate displacement and velocity
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        dz = positions[-1][2] - positions[-2][2]
        dt = 1  # Assuming 1 second between frames

        return np.sqrt(dx**2 + dy**2 + dz**2) / dt

    def detect_fish(self):
        """Main detection and tracking loop"""
        while True:
            # Read frame from video
            ret, frame = self.cap.read()
            if not ret:
                break

            # Prepare frame for YOLO
            blob = cv2.dnn.blobFromImage(
                frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Process detections
            width, height = frame.shape[1], frame.shape[0]
            fish_detections = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Check if detected object is a fish with high confidence
                    if (COCO_CLASSES[class_id] == 'cell phone' and
                            confidence > 0.9):  # Increased confidence threshold

                        # Calculate bounding box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2

                        # Get distance from ultrasonic sensor
                        distance = self.get_ultrasonic_distance()

                        # Store fish detection
                        fish_detections.append({
                            'bbox': (x, y, w, h),
                            'center': (center_x, center_y),
                            'distance': distance,
                            'confidence': confidence
                        })

            # Process each detected fish
            for fish in fish_detections:
                x, y, w, h = fish['bbox']
                center_x, center_y = fish['center']

                # Track fish positions
                self.fish_positions.append(
                    (center_x, center_y, fish['distance'])
                )

                # Draw bounding box for fish
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add label with confidence
                label = f"Fish ({fish['confidence']:.2f})"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate and display velocity if multiple positions tracked
            if len(self.fish_positions) >= 2:
                velocity = self.calculate_velocity(self.fish_positions)
                cv2.putText(frame, f"Velocity: {velocity:.2f} m/s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            # Display frame
            cv2.imshow("Fish Detection", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.ultrasonic_serial:
            self.ultrasonic_serial.close()


# Run the fish tracker
if __name__ == "__main__":
    tracker = FishTracker('./yolov3.weights', './yolov3.cfg')
    tracker.detect_fish()
