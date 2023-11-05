
import cv2 
import numpy as np
import math
import csv

def save_data_to_csv(image_name, data, csv_file_name):
    # Prepare the header for the CSV file
   

    # Write the data to the CSV file
    with open(csv_file_name, mode='a+', newline='') as file:
        writer = csv.writer(file)
        stri =""
        # Append the image name to each row
        for i in (data):
            for j in i:
                stri+=str(j)+","
        stri+=image_name
        # print(stri.split(","))
        writer.writerow(stri.split(","))       
def dis(x,d=0):
    if x>0:
        if x < 815:
            f= (815-0)/10
            d= d + x/f
        else:
            d= d+10  
            
    if  x > 815:
        if x < 1059:
            f= (1059-815)/10
            d=d+(x-815)/f
        else:
            d= d+10 
        
    if  x > 1059:
        if x < 1175:
            f= (1175-1059)/10
            d=d+(x-1059)/f
        else:
            d= d+10 
        
    if  x > 1175:
        if x < 1245:
            f= (1245-1175)/10
            d=d+(x-1175)/f
        else:
            d= d+10 
       
    if x >1245:
        if x < 1282:
            f= (1282-1245)/10
            d=d+(x-1245)/f
        else:
            d= d+10 
        
    if  x > 1282:
        if x < 1313:
            f= (1313-1282)/10
            d=d+(x-1282)/f
        else:
            d= d+10 
       
    if x > 1313:
        if x < 1331:
            f= (1331-1313)/10
            d=d+(x-1313)/f
        else:
            d= d+10 
        
    if x > 1331:
        if x < 1345:
            f= (1345-1331)/10
            d=d+(x-1331)/f
        else:
            d= d+10 
        
    if x > 1345:
        if x < 1355:
            f= (1355-1345)/10
            d=d+(x-1345)/f
        else:
            d= d+10 
        
    if x > 1355:
        if x < 1360:
            f= (1360-1355)/10
            d=d+(x-1355)/f
        else:
            d= d+10 
    
    return d

def calculate_angle_between_points(x1, y1, x2, y2, x3, y3):
    # Define the three coordinates (points)
    point1 = np.array([x1, y1])
    point2 = np.array([x2, y2])
    point3 = np.array([x3, y3])

    # Calculate the vectors between the points
    vector1 = point1 - point2
    vector2 = point3 - point2

    # Calculate the magnitudes (lengths) of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)

    # Calculate the cosine of the angle between the vectors using the law of cosines
    cosine_theta = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians
    theta_radians = np.arccos(cosine_theta)

    # Convert the angle from radians to degrees and round it to the nearest integer
    theta_degrees = (theta_radians)

    return theta_degrees

# Example usage:

center = 2020 # define observer


def click_event(event, x, y, flags, params): 
    
    
    if event == cv2.EVENT_LBUTTONDOWN: 

        global data
        # print(x, ' ', y) 
        angle_degrees = calculate_angle_between_points(x, y, center, 0, center, y)
        # print("Angle between the three points (in degrees):", angle_degrees)
        print(dis(y)/math.cos(angle_degrees))
        
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', img)
        
        data.append([x, y, dis(y)/math.cos(angle_degrees)])       
         
    if event==cv2.EVENT_RBUTTONDOWN: 

         
        print(x, ' ', y) 

         
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (x,y), font, 1, 
                    (255, 255, 0), 2) 
        cv2.imshow('image', img) 

 

     
img = cv2.imread('/Users/kaushal79/Downloads/G0035087.JPG', 1) 
data = []
     
cv2.imshow('image', img) 

cv2.setMouseCallback('image', click_event) 
while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

image_name = 'G0035087.JPG'
csv_file_name = 'data.csv'
save_data_to_csv(image_name, data, csv_file_name)
cv2.destroyAllWindows()

