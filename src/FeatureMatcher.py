import cv2 as cv

class FeatureMatcher:
    """
    This is currently just a wrapper class for OpenCV's BFMatcher class
    """
    def __init__(self, normType = cv.NORM_L2, crossCheck = False):
        self._normType = normType
        self._crossCheck = crossCheck
        self._matches = []

        self._matcher = cv.BFMatcher(self._normType, self._crossCheck)


    def findMatches(self, queryFeatures, imageFeatures):
        """
        This is just a wrapper for OpenCV's BFMatcher match function
        It takes as inputs the features for keypoints in two images,
        and returns a collection of OpenCV DMatch objects
        storing the indices of the matches in each image,
        and the distance between the matches

        TODO:
        1. add support for other optional parameters available in OpenCV
        2. add support for multiple images at once
        3. Replace OpenCV's DMatch objects with custom built versions
        """
        self._matches = self._matcher.match(queryFeatures, imageFeatures)
        return self._matches
