# import 

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

def binary_image(undistored_image, sobel_threshold, sobelx_threshold):

    # make a copy of the image 
    image = np.copy(undistored_image)
    
    # Convert to HLS space then separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    v_channel = hls[:,:,0] 
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel in x direction 
    raw_sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(raw_sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    binary_sobelx = np.zeros_like(scaled_sobelx)
    binary_sobelx[(scaled_sobelx >= sobelx_threshold[0]) & (scaled_sobelx <= sobelx_threshold[1])] = 1
    
    # Threshold and stack in the color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sobel_threshold[0]) & (s_channel <= sobel_threshold[1])] = 1
    color_binary = np.dstack(( np.zeros_like(binary_sobelx), binary_sobelx, s_binary)) * 255
    
    # Combine the two binary thresholds
    binary_image = np.zeros_like(binary_sobelx)
    binary_image[(s_binary == 1) | (binary_sobelx == 1)] = 1
    return color_binary, binary_image 
