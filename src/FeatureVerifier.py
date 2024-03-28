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
        self._sampleFMatrix
        self._fMatrix = []

    def verifyMatches(self):
        matchesLength = self._matches.length()
        sampleCount = 0
        N = float('inf')
        bestProp = 0
        # RANSAC algorithm
        while sampleCount < N:
            sampleMatrix = rnd.sample(population=self._matches,k=self._sample)
            self._sampleFMatrix = self.generateFMatrix(sampleMatrix)
            inlierMatrix = self.assessInliers()
            prop = inlierMatrix.length()/matchesLength
            invProp = 1 - prop
            if prop >= bestProp:
                bestProp = prop
                self._fMatrix = self._sampleFMatrix
                self._inlierMatrix = inlierMatrix
            N = np.log(0.01)/np.log((1-((1-invProp)^self._sample)))
            sampleCount += 1
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
    
    def generateFMatrix(self,sampleMatrix):
        coeffMatrix = []
        fMatrix = []
        zeroRow = np.zeros(9)
        normalizer = [nrm.Normalizer(),nrm.Normalizer()]
        img1Normalized = normalizer[0].normalizeMatrix(sampleMatrix[:][0])
        img2Normalized = normalizer[1].normalizeMatrix(sampleMatrix[:][1])
        # To determine the fundamental matrix, we must step through the following:
        # 1c. Find min value in sigma, then select that same row in V transpose. Create a 3x3
        #     matrix from the 9-entry vector from V transpose, F.
        # 1d. Do SVD on F, then set the third singular value to 0. Multiply resultant matrices
        #     together to get F'. This is our final matrix for this sample.
        # 3. Return the result and continue the main
        #  verification algorithm.
        # To generate our solution matrix: for all points in the matrix:
        for i in range(0,self._sample):
            # 1. Generate entries for the 8 necessary equations, based on the formulae given.
            x_norm = np.atleast2d(img1Normalized[i])
            y_norm = np.atleast2d(img2Normalized[i]).T
            coeffMatrix.append(np.flatten(np.matmul(x_norm, y_norm)))
        for i in range(0,self._sample-7):
            # 1b. Add k - 7 rows of 0s to F to get A then perform SVD on A = U*sigma*V transpose. 
            coeffMatrix.append(zeroRow)
        coeffSVD = np.svd(coeffMatrix)
        # TODO: find minimum singular value and assoc. column in Vh to reconstruct our first F matrix.
        # After all equations are appended, append self._sample-7 rows of 0s to fMatrix, and return.
        return fMatrix
    
    # TODO: Ensure that distance is correct
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
            result = np.matmul(np.matmul(x_point,self._sampleFMatrix),y_point)
            if np.abs(result) < threshold:
                inlierMatrix.append(match)
        return inlierMatrix