# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:52:05 2022

@author: Cecilia

Camera calibration information needed
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


minDisparity = 0
# in some implementations, numDisparity must be dividable by 16
numDisparities = 30 - minDisparity
blockSize = 5
uniquenessRatio = 5
# normally a value within the 5 - 15 range is good enough
speckleWindowSize = 3
speckleRange = 3
disp12MaxDiff = 200
# set to non-positive value to disable the check
P1 = 600
# 8 * number_of_image_channels * SADWindowSize * SADWindowSize
P2 = 2400
# the larger P2 is, the smoother the disparity
# 32 * number_of_image_channels * SADWindowSize * SADWindowSize

stereo = cv.StereoSGBM_create(
    minDisparity = minDisparity,
    numDisparities = numDisparities,
    blockSize = blockSize,
    uniquenessRatio = uniquenessRatio,
    speckleRange = speckleRange,
    speckleWindowSize = speckleWindowSize,
    disp12MaxDiff = disp12MaxDiff,
    P1 = P1,
    P2 = P2
)

imgL = cv.imread('F:/Arbeit/cureVision/cali/wound/uL.png')
imgR = cv.imread('F:/Arbeit/cureVision/cali/wound/uR.png')
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
blurredL = cv.GaussianBlur(imgL,(7, 7), cv.BORDER_DEFAULT)
blurredR = cv.GaussianBlur(imgR, (7, 7), cv.BORDER_DEFAULT)

disparity = stereo.compute(blurredL, blurredR)

plt.imshow(disparity, 'gray')
plt.show()


