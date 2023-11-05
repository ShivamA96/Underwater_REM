import cv2
import os
import math

coordinates = {}
counter = 0
max_width = 1920
max_height = 1080
image_paths = []
origin = None
axis_points = []
black_bands = []
pixel_to_cm_conversion = {}

def get_mouse_coordinates(event, x, y, flags, param):
    global counter
    global coordinates
    global origin
    global axis_points
    global black_bands

    if event == cv2.EVENT_LBUTTONDOWN:
        if counter < len(black_bands):
            black_bands.append((x, y))
            print(f"Black band {counter+1} clicked at (x, y): ({x}, {y})")
        elif counter == len(black_bands):
            axis_points.append((x, y))
            print(f"Axis point {counter+1} set at (x, y): ({x}, {y})")
        else:
            coordinates[counter] = (x, y)
            print(f"Point {counter} clicked at (x, y): ({x}, {y})")
            distance = calculate_real_distance(x, y)
            print(f"Real estimated distance from axis: {distance} cm\n")

        counter += 1

def calculate_real_distance(x, y):
    global axis_points
    global pixel_to_cm_conversion

    x1, y1 = axis_points[0]
    x2, y2 = axis_points[1] if len(axis_points) > 1 else axis_points[0]

    angle = math.atan2(y - y1, x - x1) - math.atan2(y2 - y1, x2 - x1)
    pixel_distance = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    band = int((y - y1) / (y2 - y1))
    conversion_factor = pixel_to_cm_conversion.get(band, 0)

    real_distance = pixel_distance * math.cos(angle) * conversion_factor

    return real_distance

def reset_origin_and_points():
    global counter
    global coordinates
    global origin
    global axis_points
    global black_bands
    global pixel_to_cm_conversion

    counter = 0
    coordinates = {}
    origin = None
    axis_points = []
    black_bands = []
    pixel_to_cm_conversion = {}

def calculate_pixel_to_cm_conversion(image):
    global black_bands
    global pixel_to_cm_conversion

    for band in range(len(black_bands)):
        x1, y1 = black_bands[band]
        x2, y2 = black_bands[band + 1] if band < len(black_bands) - 1 else axis_points[0]

        band_length_pixels = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        band_length_cm = 10  # Known length of the band in centimeters

        conversion_factor = band_length_cm / band_length_pixels
        pixel_to_cm_conversion[band] = conversion_factor

for dirnames, _, filenames in os.walk("images"):
    for filename in filenames:
        image_path = os.path.join(dirnames, filename)
        image_paths.append(image_path)

for image_path in image_paths:
    reset_origin_and_points()
    image = cv2.imread(image_path)
    image = cv2.resize(image, (max_width, max_height))

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", get_mouse_coordinates)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("f"):
            break

        if counter < len(black_bands):
            black_bands.append(None)

    calculate_pixel_to_cm_conversion(image)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
