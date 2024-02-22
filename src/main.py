import cv2 as cv
import FeatureSelector as fs
import FeatureMatcher as fm
import FeatureVerifier as fv

selector = fs.FeatureSelector()
matcher = fm.FeatureMatcher()
verifier = fv.FeatureVerifier()

# images/sample.png does not exist yet
# TODO: replace with an interactive version allowing one to upload images
image = cv.imread("images/sample.png")
grayImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
image2 = cv.imread("images/sample2.png")
grayImage2 = cv.cvtColor(image2,cv.COLOR_BGR2GRAY)

keypoints1 = selector.selectKeypoints(grayImage)
features1 = selector.computeFeatures(grayImage,keypoints1)

keypoints2 = selector.selectKeypoints(grayImage2)
features2 = selector.computeFeatures(grayImage2,keypoints2)

matches = matcher.findMatches(features1, features2)

# should give a list of tuples ((x1,y1), (x2,y2)) where (x1,y1) is a point in the first image, and (x2,y2) the matching
# point in the second image
pointMatches = [(keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt) for match in matches]

'''
TODO: Add further steps in the SfM process:
(IN PROCESS) Geometric Verification: Fundamental matrix is estimated, and inlier points are 
assessed, in preparation for reconstruction.
Initialization: Optimal image pair is selected from given list of images, and the first iteration of
the reconstructed map is created.
Image registration: A new image is added to the cache. New image points are added to the image list, and 
the map is correspondingly updated.
Triangulation: Existing image points, if visible from the new image, are updated with additional redundant
information, to improve the stability of their position.
Bundle adjustment: Errors introduced in image registration and triangulation are corrected for by the
minimization of a loss function.
'''