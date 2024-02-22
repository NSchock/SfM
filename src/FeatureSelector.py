import cv2 as cv
import numpy as np

class FeatureSelector:
    """
    This is currently just a wrapper class for OpenCV's SIFT class.
    """
    def __init__(self,
                 nFeatures=0, nOctaveLayers=3, contrastThreshold=0.04,
                 edgeThreshold=10, sigma=1.6, enablePreciseUpscale=False):
        """
        For an explanation of the above parameters see the OpenCV SIFT class documentation
        """
        self._nFeatures = nFeatures
        self._nOctaveLayers = nOctaveLayers
        self._contrastThreshold = contrastThreshold
        self._edgeThreshold = edgeThreshold
        self._sigma = sigma
        self._enablePreciseUpscale = enablePreciseUpscale

        self._sift = cv.SIFT_create(self._nFeatures, self._nOctaveLayers, self._contrastThreshold,
                                      self._edgeThreshold, self._sigma, self._enablePreciseUpscale)

    def selectKeypoints(self, image, mask=None):
        """
        A wrapper function for OpenCV SIFT's detect function.
        TODO: add support for multiple images at once (OpenCV can already do this)
        """
        return self._sift.detect(image,mask)

    def getKeypointsTuples(self, image, mask=None):
        return [keypoint.pt for pt in self.selectKeypoints(image,mask)]

    def computeFeatures(self, image, keypoints):
        """
        A wrapper function for OpenCV SIFT's compute function.
        Returns a list of arrays of size 128, one for each keypoint, describing the corresponding feature

        If no keypoints have been selected yet, then this calls selectKeypoints to get keypoints.
        Thus one can just call computeFeatures(image,mask) to get both keypoints and features.
        This is different from OpenCV's functionality, where the above would be handled by detectAndCompute.

        TODO: add support for multiple images at once (OpenCV can already do this)
        """
        if not keypoints:
            keypoints = self.selectKeypoints(image)
        return self._sift.compute(image,keypoints)


