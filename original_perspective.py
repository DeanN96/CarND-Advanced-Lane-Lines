# Get imports

import numpy as np
import cv2
import glob
import math 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_original_perspective(image, top_margin, bottom_margin):
    
    #make copy of image
    img = np.copy(image)
    
    #get image size
    y_dir = img.shape[0]
    x_dir = img.shape[1]
    
    mid_x = x_dir//2
    top_y = 2*y_dir//3
    
    src = np.float32([
        (mid_x-top_margin, top_y),
        (mid_x+top_margin, top_y),
        (mid_x+bottom_margin, y_dir),
        (mid_x-bottom_margin, y_dir)
    ])   
    
    dst = np.float32([
        (mid_x-bottom_margin, 0),
        (mid_x+bottom_margin, 0),
        (mid_x+bottom_margin, y_dir),
        (mid_x-bottom_margin, y_dir)
    ])    
    
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    #reverse
    unwarped = cv2.warpPerspective(img, Minv, (x_dir, y_dir), flags=cv2.INTER_LINEAR)
    
    return unwarped
    
   
    
    
    
    
    


