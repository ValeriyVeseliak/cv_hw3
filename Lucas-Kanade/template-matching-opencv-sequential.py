# SSD - CV_TM_SQDIFF
# NCC - CV_TM_CCORR_NORMED
# SAD - CV_TM_CCOEFF
import numpy as np
import cv2 as cv
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', dest='images_folder', type=str)
parser.add_argument('--template', dest='template', type=str)
parser.add_argument('--method', dest='method', type=str)
args = parser.parse_args()

if args.method != 'NCC' and args.method != 'SSD':
    raise Exception("Unknown method")

if args.method == 'NCC':
    match_method = cv.TM_CCORR_NORMED
elif args.method == 'SSD':
    match_method = cv.TM_SQDIFF


elements = sorted(glob.glob(args.images_folder + "*.jpg"))
images = [cv.imread(file) for file in elements]
templ = cv.imread(args.template, cv.IMREAD_COLOR).astype(np.uint8)


for img_display in images:
    result = cv.matchTemplate(img_display, templ, match_method)
    cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
    if match_method == cv.TM_SQDIFF:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    cv.rectangle(img_display, matchLoc, (matchLoc[0] + templ.shape[1], matchLoc[1] + templ.shape[0]), (0, 0, 0), 2, 8, 0)
    cv.imshow("Result", img_display)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break