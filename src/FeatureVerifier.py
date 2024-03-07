import cv2 as cv
import numpy as np
import Normalizer as nrm
import random as rnd

class FeatureVerifier:

    def __init__(self,
                matches,
                sample=8,
                threshold=0.5):
        self._matches = matches
        self._threshold = threshold
        self._sample = sample
        self._inlierMatrix = []
        self._fMatrix = []

    def verifyMatches(self):
        n = self._matches.length()
        while self._fMatrix == []:
            randomIndex = [rnd.sample(population=self._matches,k=self._sample)]
            sampleMatrix = [self._matches[i] for i in randomIndex]
            self._fMatrix = self.generateFMatrix(sampleMatrix)
            inlierMatrix = self.assessInliers()
            prop = inlierMatrix.length()/n
            if prop >= self._threshold:
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
        solutionMatrix = []
        fMatrix = []
        normalizer = [nrm.Normalizer(),nrm.Normalizer()]
        img1Normalized = normalizer[0].normalizeMatrix(sampleMatrix[:][0])
        img2Normalized = normalizer[1].normalizeMatrix(sampleMatrix[:][1])
        # To determine the fundamental matrix, we must step through the following:
        # 1. Generate entries for the 8 necessary equations, based on the formulae given
        # 2. Solve this matrix for the fundamental matrix (via scipy or numpy)
        # 3. Return the result and continue the main verification algorithm.
        # To generate our solution matrix: for all points in the matrix:
        for i in range(0,sampleMatrix.length()):
            pass # Placeholder
            # Multiply img1[i], img2[i] on a 3x3 matrix of 1s to get an equation, and append to
            # fMatrix
        # After all equations are appended, append self._sample - 7 rows of 0s to fMatrix, and return.
        return fMatrix
    
    def assessInliers(self, threshold=0.5):
        inlierMatrix = []
        result = 0
        for match in self._matches:
            x_point = match[0]
            y_point = match[1]
            x_point.append(1)
            y_point.append(1)
            # Docs indicate matmul() is tolerant to no transpose and will make the necessary
            # dim adjustments itself.
            result = np.matmul(np.matmul(x_point,self._fMatrix),y_point)
            if np.abs(result) < threshold:
                inlierMatrix.append(match)
        return inlierMatrix