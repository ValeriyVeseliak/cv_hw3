import numpy as np
import cv2 as cv
import glob
import argparse


def cam_center(points):
    x = np.float32((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0)
    y = np.float32((points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)


def mean_center(points):
    print(points)
    x = np.float32((points[0][0] + points[1][0]) / 2.0)
    y = np.float32((points[0][1] + points[1][1]) / 2.0)
    print(x,y)
    return np.array([np.float32(x), np.float32(y)], np.float32)


parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', dest='images_folder', type=str, help='path to image file')
parser.add_argument('--roi_path', dest='roi_path', type=str, help='path to image file')
args = parser.parse_args()

elements = sorted(glob.glob(args.images_folder + "*.jpg"))
images = [cv.imread(file) for file in elements]
start_frame = images[0].astype(np.uint8)
roi = cv.imread(args.roi_path).astype(np.uint8)
h, w, c = roi.shape
# retrieving  track window by cropped part of image
match = cv.matchTemplate(start_frame, roi, cv.TM_CCOEFF_NORMED)
threshold = 0.8
position = np.where(match >= threshold)
point = list(zip(*position[::-1]))[1]
x, y, w, h = point[0], point[1], w, h
track_window = (x, y, w, h)
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
# mask = cv.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))#dog

roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

kalman = cv.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

measurement = np.array((2, 1), np.float32)
prediction = np.zeros((2, 1), np.float32)

for frame in images:
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # CamShift algorithm approach
    # -- works very well for CamShift and remove fluctuations of window
    # -- also add stability for Camshift

    # ret, track_window = cv.CamShift(dst, track_window, term_crit)
    # x, y, w, h = track_window
    # pts = cv.boxPoints(ret)
    # pts = np.int0(pts)
    # kalman.correct(cam_center(pts))
    # prediction = kalman.predict()
    # img2 = cv.rectangle(frame, (prediction[0] - (0.5 * w), prediction[1] - (0.5 * h)),
    #                     (prediction[0] + (0.5 * w), prediction[1] + (0.5 * h)), (0, 255, 0), 2)

    # MeanShift algorithm
    ret, track_window = cv.meanShift(dst, track_window, term_crit)
    x, y, w, h = track_window
    pts = np.array([[x, y], [x + w, y + h]])
    rect_center = mean_center(pts)
    kalman.correct(rect_center)
    prediction = kalman.predict()
    img2 = cv.rectangle(frame, (int(prediction[0] - (0.5 * w)), int(prediction[1] - (0.5 * h))),
                        (int(prediction[0] + (0.5 * w)), int(prediction[1] + (0.5 * h))), 100, 2)

    # simple MeanShift code
    # For comparison reason added blue as original MeanShift and darker one as smoothed by Kalman filter
    img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

    cv.imshow('img2', img2)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
