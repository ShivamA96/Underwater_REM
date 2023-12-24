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
    complete_arr = []
    for coord_set in coordinates:
        inside_flags = []  # Flag array to track inside/outside for each particle
        for coord in coord_set:
            point = np.array(coord)
            inside = is_inside_cone(point)
            inside_flags.append(inside)
        complete_arr.append(inside_flags)
    complete_arr = np.array(complete_arr).flatten()
    for i in range(len(complete_arr) - 1):                
                if i == len(complete_arr) - 10:
                    break
                current_flags = complete_arr[i]
                # print(current_flags)
                next_flags = complete_arr[i + 10]
                # print(next_flags)
                # Find where the transition occurs from False to True for each pair
                transition_points = np.logical_and(~current_flags, next_flags)
                # Increment encounter count for each transition point in the pair
            
                encounter_count += np.sum(transition_points)


# Count encounters for all coordinate sets
count_encounters(coordinates)
print(f"Total encounters: {encounter_count}")



# if current flag = true and above this if prev flag = false then count +1 data mein difference should of 10 and not 1

# segment data in 3 dimensional array

# 

# y = d * r0 [(r(pi * r + root(h**2)))] * v (vmed) * t (total time 1800s) 


# y = number of encounters


# r of cone = 0.25
# h = 1

# calculate L

# r0 = surface area of cone

