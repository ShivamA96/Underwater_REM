import RPi.GPIO as GPIO
import time
import numpy as np
import cv2


class UltrasonicCameraMapper:
    def __init__(self, trigger_pin, echo_pin, camera_matrix, sensor_height, sensor_angle):
        """
        Initialize ultrasonic sensor and mapping parameters

        Args:
            trigger_pin (int): GPIO pin number for trigger
            echo_pin (int): GPIO pin number for echo
            camera_matrix (np.array): Camera intrinsic matrix
            sensor_height (float): Height of ultrasonic sensor above reference plane
            sensor_angle (float): Angle of sensor relative to camera's optical axis
        """
        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin

        GPIO.setup(trigger_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)

        # Camera Mapping Parameters
        self.camera_matrix = camera_matrix
        self.sensor_height = sensor_height  # in meters
        self.sensor_angle = np.radians(sensor_angle)  # convert to radians

        # Calibration points storage
        self.calibration_points = []

    def measure_distance(self):
        """
        Measure distance using ultrasonic sensor

        Returns:
            float: Distance in meters
        """
        # Ensure trigger is LOW
        GPIO.output(self.trigger_pin, False)
        time.sleep(0.1)

        # Send 10us pulse to trigger
        GPIO.output(self.trigger_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trigger_pin, False)

        # Wait for echo start
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()

        # Wait for echo end
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()

        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound is 343 m/s, divided by 2
        distance = round(distance / 100, 2)  # Convert to meters

        return distance

    def map_sensor_to_camera(self, sensor_distance):
        """
        Map ultrasonic sensor distance to camera frame coordinates

        Args:
            sensor_distance (float): Distance measured by ultrasonic sensor

        Returns:
            tuple: Estimated (x, y) pixel coordinates in camera frame
        """
        # Calculate world coordinates
        world_x = sensor_distance * np.cos(self.sensor_angle)
        world_y = sensor_distance * np.sin(self.sensor_angle)
        world_z = self.sensor_height

        # Project to camera pixel coordinates
        pixel_coords = cv2.projectPoints(
            np.array([(world_x, world_y, world_z)]),
            np.zeros(3),
            np.zeros(3),
            self.camera_matrix,
            None
        )[0][0][0]

        return pixel_coords

    def add_calibration_point(self, sensor_distance, pixel_coordinates):
        """
        Add a calibration point for sensor-camera mapping

        Args:
            sensor_distance (float): Measured sensor distance
            pixel_coordinates (tuple): Corresponding pixel coordinates
        """
        self.calibration_points.append((sensor_distance, pixel_coordinates))

    def run_distance_mapping_demo(self, num_measurements=10):
        """
        Demonstrate distance measurement and camera mapping

        Args:
            num_measurements (int): Number of distance measurements to take
        """
        try:
            for _ in range(num_measurements):
                # Measure distance
                distance = self.measure_distance()

                # Map to camera coordinates
                pixel_coords = self.map_sensor_to_camera(distance)

                print(
                    f"Distance: {distance} m | Pixel Coordinates: {pixel_coords}")
                time.sleep(1)  # Delay between measurements

        except KeyboardInterrupt:
            print("\nMeasurement stopped by user")

        finally:
            # Clean up GPIO on exit
            GPIO.cleanup()


def main():
    # Example camera matrix (needs to be calibrated for specific camera)
    camera_matrix = np.array([
        [1000, 0, 640],    # fx, skew, cx
        [0, 1000, 360],    # 0, fy, cy
        [0, 0, 1]          # perspective transformation
    ])

    # Initialize mapper with GPIO pins and camera parameters
    mapper = UltrasonicCameraMapper(
        trigger_pin=23,        # Example GPIO pin for trigger
        echo_pin=24,           # Example GPIO pin for echo
        camera_matrix=camera_matrix,
        sensor_height=0.5,     # 50 cm above reference plane
        sensor_angle=30        # 30 degrees from vertical
    )

    # Optional: Add calibration points
    mapper.add_calibration_point(1.0, (300, 200))
    mapper.add_calibration_point(2.0, (500, 350))
    mapper.add_calibration_point(3.0, (650, 450))

    # Run distance mapping demonstration
    mapper.run_distance_mapping_demo()


if __name__ == "__main__":
    main()
