import cv2 as cv
import numpy as np

class Normalizer:

    def __init__(self,
                translation=[],
                scalar=1,
                normalized=[]):
        self._translation = translation
        self._scalar = scalar
        self._normalized = normalized


    def normalizeMatrix(self,initial):
        n = initial.length()
        sumVector = [0]*2
        # Summing all points to find averages of x and y
        for i in initial:
            sumVector = i + sumVector
        self._translation = sumVector/n
        translatedMatrix = [[0,0]]*n
        # Subtracting translation matrix to move the centroid (avg) to (0,0)
        for i in range(0,n):
            translatedMatrix[i] = initial[i] - self._translation
        translatedLengthSum = 0
        # Summing all lengths of points in translated matrix
        for i in range(0,n):
            translatedLengthSum += np.sqrt(((translatedMatrix[i][0])^2)+((translatedMatrix[i][1])^2))
        self._scalar = translatedLengthSum/(n*np.sqrt(2))
        self._normalized = translatedMatrix/self._scalar
        # Return normalized vector
        return self._normalized
    
    def denormalizeMatrix(self, normal):
        denormalizedMatrix = []
        for i in normal:
            denormalizedMatrix.append((i*self._scalar)+self._translation)
        return denormalizedMatrix
