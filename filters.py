# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:47:14 2022

Just some basic filters written by hand.

@author: Cecilia
"""

import cv2 as cv
import numpy as np
import utils


def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
    # we allow kernel sizes to be specified as arguments for this function
    # if performance problems encountered:
    # try decreasing blurKsize value
    if blurKsize >= 3:
        blurredSrc = cv.medianBlur(src, blurKsize)
        graySrc = cv.cvtColor(blurredSrc, cv.COLOR_BGR2GRAY)
    else:
        graySrc = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.Laplacian(graySrc, cv.CV_8U, graySrc, ksize = edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
        cv.merge(channels, dst)   

class VConvolutionFilter(object):
    ''' a filter that applies a convolution to V'''
    def __init__(self, kernel):
        self._kernel = kernel
    def apply(self, sec, dst):
        ''' apply the filter with a BGR or gray source/destination'''
        cv.filter2D(src, -1, self._kernel, dst)

class SharpenFilter(VConvolutionFilter):
    ''' a sharpen filter with a 1-pixel radius'''
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)
        # weights sum up to be 1
        # leaves the overall brightness unchanged

class FindEdgeFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)
        
class BlurFilter(VConvolutionFilter):
    ''' a blur filter with a 2-pixel radius'''
    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)
        
class EmbossFilter(VConvolutionFilter):
    ''' an emboss filter with a 1-pixel radius'''
    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])
        VConvolutionFilter.__init__(self, kernel)
