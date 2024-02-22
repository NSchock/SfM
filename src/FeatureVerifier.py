import cv2 as cv
import numpy as np

class FeatureVerifier:

    def __init__(self,
                matches,
                threshold=0.5):
        self._matches = matches
        self._threshold = threshold
        self._fMatrix = []

    def verifyMatches(self):
        n = self._matches.length()
        while self._fMatrix == []:
            randomIndex = [np.randInt(0,n,8)]
            sampleMatrix = [self._matches[i] for i in randomIndex]
        # 1. Start loop (iterate until sufficient confidence in fundamental matrix)
        # 2. Pick a random sample from the given matches.
        # 3. Calculate candidate fundamental matrix that works wrt to the random sample.
        # 4. Use that matrix to assess the viability of the rest of the points and 
        #    record inlier proportion.
        # 5. Compare proportion to threshold:
        #   a. If proportion < threshold, discard matrix and iterate to step 2.
        #   b. If proportion >= threshold, save matrix as self._fMatrix, and exit.
        return self._fMatrix