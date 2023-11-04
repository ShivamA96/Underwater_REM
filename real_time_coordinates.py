import cv2
import os
import pandas as pd
import math

coordinates = {}
counter = 0
max_width = 1920
max_height = 1080
image_paths = []
origin = None

def get_mouse_coordinates(event, x, y, flags, param):
    global counter
    global coordinates
    global origin

    if event == cv2.EVENT_LBUTTONDOWN:
        if counter == 0:
            origin = (x, y)
            print(f"Origin point set at (x, y): ({x}, {y})")
        else:
            coordinates[counter] = (x, y)
            print(f"Point {counter} clicked at (x, y): ({x}, {y})")
            distance = math.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2) # calculate the distance from the origin point
            print(f"Distance from origin: {distance} pixels\n")

        counter += 1

def reset_origin_and_points():
    global counter
    global coordinates
    global origin

    counter = 0
    coordinates = {}
    origin = None

for dirnames, _, filenames in os.walk("images"):
    for filename in filenames:
        image_path = os.path.join(dirnames, filename)
        image_paths.append(image_path)

for image_path in image_paths:
    reset_origin_and_points()  # Reset origin and points for each new image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (max_width, max_height))

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", get_mouse_coordinates)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

cv2.destroyAllWindows()
