# SSD - CV_TM_SQDIFF
# NCC - CV_TM_CCORR_NORMED
import cv2 as cv
import argparse

# ------ Implementation of template matching with OpenCV ------

parser = argparse.ArgumentParser()
parser.add_argument('--image', dest='image', type=str)
parser.add_argument('--template', dest='template', type=str)
parser.add_argument('--method', dest='method', type=str)
args = parser.parse_args()

if args.method != 'NCC' and args.method != 'SSD':
    raise Exception("Unknown method")

if args.method == 'NCC':
    match_method = cv.TM_CCORR_NORMED
elif args.method == 'SSD':
    match_method = cv.TM_SQDIFF

img = cv.imread(args.image, cv.IMREAD_COLOR)
templ = cv.imread(args.template, cv.IMREAD_COLOR)

img_display = img.copy()
result = cv.matchTemplate(img, templ, match_method)
cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)

_minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)

if match_method == cv.TM_SQDIFF:
    matchLoc = minLoc
else:
    matchLoc = maxLoc

cv.rectangle(img_display, matchLoc, (matchLoc[0] + templ.shape[1], matchLoc[1] + templ.shape[0]), (0, 0, 0), 2, 8, 0)
cv.imshow("Source", img_display)
cv.waitKey(0)
