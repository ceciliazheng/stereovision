import sys
import numpy as np
import time
import cv2
from calibration import stereoCalibration

def undistorted(frameR, frameL):
    
    pathL = "F:/Arbeit/cureVision/cali/stereoLeft/*.png"
    pathR = "F:/Arbeit/cureVision/cali/stereoRight/*.png"
    cameraMatrixL, cameraMatrixR, retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = stereoCalibration(pathL, pathR)

    hR,wR = frameR.shape[:2]
    hL,wL = frameL.shape[:2]

    #Undistort images
    frame_undistortedR = cv2.undistort(frameR, cameraMatrixR, distR, None, newCameraMatrixR)
    frame_undistortedL = cv2.undistort(frameL, cameraMatrixL, distL, None, newCameraMatrixL)

    ##Uncomment if you want help lines:
    #frame_undistortedR = cv2.line(frame_undistortedR, (0,int(hR/2)), (wR,240), (0, 255, 0) , 5)
    #frame_undistortedR = cv2.line(frame_undistortedR, (int(wR/2),0), (int(wR/2),hR), (0, 255, 0) , 5)
    #frame_undistortedL = cv2.line(frame_undistortedL, (int(wL/2),0), (int(wL/2),hL), (0, 255, 0) , 5)
    #frame_undistortedL = cv2.line(frame_undistortedL, (0,int(hL/2)), (wL,240), (0, 255, 0) , 5)

    return frame_undistortedR, frame_undistortedL

if __name__ == '__main__':
    path = "F:/Arbeit/cureVision/cali/wound/"
    
    frameR = cv2.imread(path + "right.png")
    frameL = cv2.imread(path + "left.png")
    
    undisR, undisL = undistorted(frameR, frameL)
    cv2.imwrite("uR.png", undisR)
    cv2.imwrite("uL.png", undisL)