# import 

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_colored_area(perspective_transformed_image, margin, nwindows, minpixels):
    
    # make copy of the image 
    image = np.copy(perspective_transformed_image)
    y_dir = image.shape[0]
    x_dir = image.shape[1]

    # Grab non zero pixels
    nonzero = image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Take a histogram for the bottom half of the image
    histogram = np.sum(image[y_dir//2:,:], axis=0)
    
    # draw and visualize
    out_image = np.dstack((image, image, image))*255
    
    # Find the peaks of the histogram
    mid = np.int(histogram.shape[0]//2)
    leftx_hist = np.argmax(histogram[:mid])
    rightx_hist = np.argmax(histogram[mid:]) + mid

    # Set height for windows 
    window_height = np.int(y_dir//nwindows)

    # Create empty lists to receive left and right lane pixel indices
    left_lane = []
    right_lane = []

    # Step through windows
    for window in range(nwindows):
        
        # Identify window boundaries
        win_y_low = y_dir - (window+1)*window_height
        win_y_high = y_dir - window*window_height
        win_x_left_low = leftx_hist - margin
        win_x_left_high = leftx_hist + margin
        win_x_right_low = rightx_hist - margin
        win_x_right_high = rightx_hist + margin
        
        # Identify the nonzero pixels 
        good_left_candidates = ((nonzero_y >= win_y_low) 
                                & (nonzero_y < win_y_high) 
                                & (nonzero_x >= win_x_left_low) 
                                &  (nonzero_x < win_x_left_high))
        
        good_left = good_left_candidates.nonzero()[0]
        
        good_right_candidates = ((nonzero_y >= win_y_low) 
                                 & (nonzero_y < win_y_high) 
                                 & (nonzero_x >= win_x_right_low) 
                                 &  (nonzero_x < win_x_right_high))
        
        good_right = good_right_candidates.nonzero()[0]
        
        # Append
        left_lane.append(good_left)
        right_lane.append(good_right)
        
        # recenter next window on their mean position if greater than minimum pixels
        if len(good_left) > minpixels:
            leftx_hist = np.int(np.mean(nonzero_x[good_left]))
            
        if len(good_right) > minpixels:        
            rightx_hist = np.int(np.mean(nonzero_x[good_right]))

    # Concatenate 
    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    # Extract pixel positions for left and right line 
    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane] 
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    # Fit a second order polynomial to each 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # values for plotting
    plot_y = np.linspace(0, y_dir-1, y_dir )
    left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
 
    # Create selection window
    window_image = np.zeros_like(out_image)
 
    # polygon for  search window area
    left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x-margin, plot_y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x+margin, plot_y])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x-margin, plot_y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x+margin, plot_y])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane
    cv2.fillPoly(window_image, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_image, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_image, 1, window_image, 0.3, 0)
    
    # Plot the polynomial line
    # plt.plot(left_fit_x, plot_y, color='yellow')
    # plt.plot(right_fit_x, plot_y, color='yellow') 
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast 
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    return color_warp
