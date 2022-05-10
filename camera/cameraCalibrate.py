# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:28:54 2022

@author: Cecilia

used to calibrate each camera individually
"""

import glob
import io
import os
import sys
import time

import cv2 as cv
import numpy as np

'''Always perform manual camera calibration.'''

x = input('Please enter number of rows of checkerboard')
y = input('Please enter number of columns of checkerboard')

print(f'The widthe of checkboard is {x} and the height is {y}.')

'''Usually a checkerboard of 6 x 9 is used.'''
CHECKERBOARD = (x,y)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objPoints = [] # 3D points in real world space
imgPoints = [] # 2D points in image plane

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# input path might need to be changed
# Extracting path of individual image stored in a given directory
images = glob.glob('./images/*.jpg')
for frame in images:
    img = cv.imread(frame)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, 
                                            cv.CALIB_CB_ADAPTIVE_THRESH + 
                                            cv.CALIB_CB_FAST_CHECK + 
                                            cv.CALIB_CB_NORMALIZE_IMAGE)
    '''
    If desired number of corners are detected,
    we refine the pixel coordinates and display them
    on the images of the checkerboard
    '''
    if ret == True:
        objPoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgPoints.append(corners2)
        # draw and display the corners
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
    cv.imshow('img', img)
    cv.waitKey(0)

cv.destroyAllWindows()
h,w = img.shape[:2]

# extrinsic matrices can be obtained by manual measurement
# or can be usually found at product description of stereo camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, 
                                                  gray.shape[::-1], None, None)

print("EMSE: \n") # RMSE value (root mean square error)
print(ret)
print("Camera matrix : \n")
print(mtx)
print('dist : \n') # distortion coefficient
print(dist)
print('rvecs : \n') # per frame rotation
print(rvecs)
print('tvecs : \n') # per frame translation
print(tvecs)