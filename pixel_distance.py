import cv2
import math

def calculate_pixel_distance(image_path, point1, point2):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    x1, y1 = int(point1[0]), int(point1[1])
    x2, y2 = int(point2[0]), int(point2[1])

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance