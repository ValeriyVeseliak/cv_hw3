import numpy as np
import cv2 as cv
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', dest='images_folder', type=str, help='path to image file')
parser.add_argument('--roi_path', dest='roi_path', type=str, help='path to image file')
args = parser.parse_args()

elements = sorted(glob.glob(args.images_folder+"*.jpg"))
images = [cv.imread(file) for file in elements]
start_frame = images[0].astype(np.uint8)
roi = cv.imread(args.roi_path).astype(np.uint8)
h,w,c = roi.shape
match = cv.matchTemplate(start_frame, roi, cv.TM_CCOEFF_NORMED)
threshold = 0.8
position = np.where(match >= threshold)
point = list(zip(*position[::-1]))[0]
x, y, w, h = point[0], point[1], w, h

track_window = (x, y, w, h)
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 120., 32.)), np.array((180., 250., 250.))) #surfer
# mask = cv.inRange(hsv_roi, np.array((50., 200., 120.)), np.array((90., 250., 250.))) # biker
# mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.))) #dragon baby
# mask = cv.inRange(hsv_roi, np.array((0., 30., 20.)), np.array((180., 255., 255.))) # dog
# mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 180., 180.))) # jump
# mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 250., 250.))) # baby

roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

for frame in images:
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    ret, track_window = cv.CamShift(dst, track_window, term_crit)
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv.polylines(frame, [pts], True, 255, 2)
    cv.imshow('Result', img2)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break