# Get imports
import numpy as np
import cv2
import glob
import math 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_transformed_perspective(image, top_margin, bottom_margin):
    
    # make a copy of the image 
    image = np.copy(image)
    
    # get image dimensions 
    y = image.shape[0]
    x = image.shape[1] 
    
    # make trapezoid 
    x_shape = image.shape[1]
    y_shape = image.shape[0]
    
    middle_x = x_shape//2
    top_y = 2*y_shape//3
     
    src = np.float32([
        (middle_x-top_margin, top_y),
        (middle_x+top_margin, top_y),
        (middle_x+bottom_margin, y_shape),
        (middle_x-bottom_margin, y_shape)
    ])

    dst = np.float32([
        (middle_x-bottom_margin, 0),
        (middle_x+bottom_margin, 0),
        (middle_x+bottom_margin, y_shape),
        (middle_x-bottom_margin, y_shape)
    ])

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Calculate top-down image using perspective transform matrix
    warped = cv2.warpPerspective(image, M, (x, y), flags=cv2.INTER_LINEAR)

    return warped
