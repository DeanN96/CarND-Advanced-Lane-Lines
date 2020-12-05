# imports 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Helper function for camera calibration

def calibration(directory):
    
    # read in images from directory 
    images = directory 
    
    # prepare object points from (0,0,0) to (0,6,9)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Storage for object points and image points
    objpoints = []
    imgpoints = []

    # search for chessboard corners
    for index, fname in enumerate(images):
        
        # current image 
        img = cv2.imread(fname)
        
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # add object points and image points if found 
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and save the corners to file 
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imwrite('./output_images/'+ 'Corners_found'+str(index)+'.jpg', img)

    # get camera calibration matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
       
    return mtx, dist