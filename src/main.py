import cv2 as cv
import FeatureSelector as fs
import FeatureMatcher as fm

selector = fs.FeatureSelector()
matcher = fm.FeatureMatcher()


# images/sample.png does not exist yet
# TODO: replace with an interactive version allowing one to upload images
image = cv.imread("images/sample.png")
grayImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
image2 = cv.imread("images/sample2.png")
grayImage2 = cv.cvtColor(image2,cv.COLOR_BGR2GRAY)

keypoints1 = selector.selectKeypoints(grayImage)
features1 = selector.computeFeatures(grayImage,keypoints)

keypoints2 = selector.selectKeypoints(grayImage2)
features2 = selector.computeFeatures(grayImage2,keypoints2)

matches = matcher.findMatches(features1, features2)

# should give a list of tuples ((x1,y1), (x2,y2)) where (x1,y1) is a point in the first image, and (x2,y2) the matching
# point in the second image
pointMatches = [(keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt) for match in matches]

