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
        translatedLengthSum = 0
        # Summing all points to find averages of x and y
        sumVector = np.sum(initial)
        self._translation = sumVector/n
        translatedMatrix = [self._translation]*n
        # Subtracting translation matrix to move the centroid (avg) to (0,0)
        translatedMatrix = np.add(initial,-translatedMatrix)
        # Summing all lengths of points in translated matrix
        translatedLengthSum = np.sum(np.sqrt(np.sum(np.square(translatedMatrix),axis=1)))
        self._scalar = translatedLengthSum/(n*np.sqrt(2))
        self._normalized = translatedMatrix/self._scalar
        # Return normalized vector
        return self._normalized
    
    def denormalizeMatrix(self, normal):
        denormalizedMatrix = []
        for i in normal:
            denormalizedMatrix.append((i*self._scalar)+self._translation)
        return denormalizedMatrix
