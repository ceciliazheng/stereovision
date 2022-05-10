# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:03:53 2022

@author: Cecilia

Stereo camera calibration.
This step should only be performed after each camera 
has been calibrated individually.

Use synched frames.
Calibration quality is related to sharpness of images.

"""

import cv2 as cv
import glob
import numpy as np
import sys

def stereoCalibrate(mtx1, dist1, mtx2, dist2, framesFolder):
    # mtx: matrix
    # dist: distorsion
    # read the synched frames
    imagesNames = glob.glob(framesFolder)
    imagesNames = sorted(imagesNames)
    c1ImageNames = imagesNames[:len(imagesNames)//2]
    c2ImageNames = imagesNames[len(imagesNames)//2:]
    c1Imgs = []
    c2Imgs = []
    for img1, img2 in zip(c1ImageNames, c2ImageNames):
        _im = cv.imread(img1, 1)
        c1Imgs.append(_im)
        
        _im = cv.imread(img2, 1)
        c2Imgs.append(_im)
        
    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    
    rows = 5
    columns = 8
    worldScale = 1
    
    objp = np.zeros((row * columns, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:columns].T.reshape(-1,2)
    objp = worldScale * objp
    
    # pixel coordinates of checkerboards
    imgPointsLeft = [] # 2D points in image plane
    imgPointsRight = []
    
    #coordinates of the checkerboard in checkerboard world space
    objPoints = [] # 3D points in real world space
    
    for frame1, frame2 in zip(c1Imgs, c2Imgs):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        GRAY2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        cRet1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
        cRet2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)
        
        if cRet1 == True and cRet2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            
            cv.drawChessboardCorners(frame1, (5,8), corners1, cRet1)
            cv.imshow('img', frame1)
            
            cv.drawChessboardCorners(frame2, (5,8), corners2, cRet2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(500)
            
            objPoints.append(objp)
            imgPointsLeft.append(corners1)
            imgPointsRight.append(corners2)
    
    '''
    R: rotational matrix: coordinate rotation matrix to go from C1 coordinate system to C2 coordinate system
    T: translation vector
    E: essential matrix
    F: fundamental matrix
    The translation vector here is the location from C2 to C1
    world coordinate to C2:
        R2 = R * R1
        T2 = RT1 + T
    '''
    stereocalibrationFlags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, 
                                                                 imgpoints_left, 
                                                                 imgpoints_right, 
                                                                 mtx1, dist1,
                                                                 mtx2, dist2, 
                                                                 (width, height), 
                                                                 criteria = criteria, 
                                                                 flags = stereocalibration_flags)
    print(ret)
    return R, T
