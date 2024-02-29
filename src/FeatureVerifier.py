import cv2 as cv
import numpy as np
import Normalizer as nrm

class FeatureVerifier:

    def __init__(self,
                matches,
                threshold=0.5):
        self._matches = matches
        self._threshold = threshold
        self._inlierMatrix = []
        self._fMatrix = []

    def verifyMatches(self):
        n = self._matches.length()
        while self._fMatrix == []:
            randomIndex = [np.randInt(0,n,8)]
            sampleMatrix = [self._matches[i] for i in randomIndex]
            sampleFMatrix = self.generateFMatrix(sampleMatrix)
            inlierMatrix = self.assessInliers(sampleFMatrix, self._matches)
            prop = inlierMatrix.length()/n
            if prop >= self._threshold:
                self._fMatrix = sampleFMatrix
                self._inlierMatrix = inlierMatrix
        # 1. Start loop (iterate until sufficient confidence in fundamental matrix) (done)
        # 2. Pick a random sample from the given matches.
        # 3. Calculate candidate fundamental matrix that works wrt to the random sample.
        # 4. Use that matrix to assess the viability of the rest of the points and 
        #    record inlier proportion and inliers.
        # 5. Compare proportion to threshold:
        #   a. If proportion < threshold, discard matrix and iterate to step 2.
        #   b. If proportion >= threshold, save matrix as self._fMatrix, and exit.
        return [self._fMatrix,self._inlierMatrix]
        # Above is placeholder for the moment, ideally it should return both matrices in a more obvious 
        # fashion.
    
    def generateFMatrix(sampleMatrix):
        normalizer = [nrm.Normalizer(),nrm.Normalizer()]
        img1Normalized = normalizer[1].normalizeMatrix(sampleMatrix[:][0])
        img2Normalized = normalizer[2].normalizeMatrix(sampleMatrix[:][1])
        # To determine the fundamental matrix, we must step through the following:
        # 1. Generate entries for the 8 necessary equations, based on the formulae given
        # 2. Solve this matrix for the fundamental matrix (via scipy or numpy)
        # 3. Return the result and continue the main verification algorithm.
        fMatrix = []
        return fMatrix