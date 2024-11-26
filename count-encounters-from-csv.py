import csv
import numpy as np

# Define cone parameters
cone_apex = np.array([0, 0, 0])  # Apex of the cone (center)
cone_direction = np.array([0, 0, 1])  # Direction of the cone
cone_angle_deg = 30  # Angle of the cone in degrees

# Global variable to count encounters
encounter_count = 0

# Function to check if a point is inside the cone
def is_inside_cone(point):
    point_vector = point - cone_apex
    point_unit_vector = point_vector / np.linalg.norm(point_vector)
    
    cos_angle = np.dot(point_unit_vector, cone_direction)
    angle_rad = np.deg2rad(cone_angle_deg)
    
    return cos_angle >= np.cos(angle_rad)

# Read coordinates from a CSV file with a custom format
coordinates = []
with open('coordinates.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    current_set = []
    for row in csv_reader:
        if row and not row[0].startswith('#'):  # Ignore lines starting with #
            coords = [float(val) for val in row]
            current_set.append(coords)
        elif current_set:  # Store coordinates set if not empty
            coordinates.append(current_set)
            current_set = []
    if current_set:  # Append the last set of coordinates if not empty
        coordinates.append(current_set)

def count_encounters(coordinates):
    global encounter_count
    for coord_set in coordinates:
        inside_flags = []  # Flag array to track inside/outside for each particle
        for coord in coord_set:
            point = np.array(coord)
            inside = is_inside_cone(point)
            inside_flags.append(inside)
        
        prev_inside = None
        for inside in inside_flags:
            if prev_inside is None:
                prev_inside = inside
            elif not prev_inside and inside:
                # Particle went from outside to inside -> Increment encounter count
                encounter_count += 1
            prev_inside = inside

# Count encounters for all coordinate sets
count_encounters(coordinates)
print(f"Total encounters: {encounter_count}")