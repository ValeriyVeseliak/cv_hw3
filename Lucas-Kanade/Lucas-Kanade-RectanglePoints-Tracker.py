import numpy as np
import cv2 as cv
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', dest='images_folder', type=str,)
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

old_gray = cv.cvtColor(start_frame, cv.COLOR_BGR2GRAY)
p0 = np.array([[[x, y]], [[x + w, y + h]]], np.float32)

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
    start_x, start_y = good_new[0].ravel()
    end_x, end_y = good_new[1].ravel()
    cv.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
