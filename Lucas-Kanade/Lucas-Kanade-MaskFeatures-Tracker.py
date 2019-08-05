import numpy as np
import cv2 as cv
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', dest='images_folder', type=str)
parser.add_argument('--roi_path', dest='roi_path', type=str)
args = parser.parse_args()

elements = sorted(glob.glob(args.images_folder + "*.jpg"))
images = [cv.imread(file) for file in elements]
roi = cv.imread(args.roi_path).astype(np.uint8)
h, w, c = roi.shape
roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

start_frame = images[0].astype(np.uint8)
match = cv.matchTemplate(start_frame, roi, cv.TM_CCOEFF_NORMED)
threshold = 0.8
position = np.where(match >= threshold)
point = list(zip(*position[::-1]))[1]
x, y, w, h = point[0], point[1], w, h

track_window = (x, y, w, h)


def create_mask(track_window, img):
    x_min, x_max = x, x + w
    y_min, y_max = y, y + h
    result_mask = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    height, width = result_mask.shape
    for i in range(height):
        for j in range(width):
            if y_max > i > y_min and x_min < j < x_max:
                result_mask[i, j] = 1
            else:
                result_mask[i, j] = 0
    return result_mask


mask = create_mask(track_window, start_frame)

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

old_gray = cv.cvtColor(start_frame, cv.COLOR_BGR2GRAY)

p0 = cv.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)

lk_params = {"winSize": (w, h), "maxLevel": 0, "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)}

color = np.random.randint(0, 255, (100, 3))
mask = np.zeros_like(start_frame)

for frame in images:
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
