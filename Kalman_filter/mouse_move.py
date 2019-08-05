import numpy as np
import cv2

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 2, 0],
                                    [0, 1, 0, 2],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.eye(4,4).astype(np.float32) * 0.03

last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)


def click_event(event, x, y, flags, param):
    global img, current_measurement, last_measurement, current_prediction, last_prediction
    if event == cv2.EVENT_MOUSEMOVE:

        last_measurement = current_measurement
        last_prediction = current_prediction

        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])

        kalman.correct(current_measurement)

        current_prediction = kalman.predict()

        last_measur_x, last_measur_y = last_measurement[0], last_measurement[1]
        curr_measur_x, curr_measur_y = x, y
        last_pred_x, last_pred_y = last_prediction[0], last_prediction[1]
        curr_pred_x, curr_pred_y = current_prediction[0], current_prediction[1]

        cv2.line(img, (last_measur_x, last_measur_y),
                 (curr_measur_x, curr_measur_y), (0, 200, 0))  # green - measurement

        cv2.line(img, (last_pred_x, last_pred_y), (curr_pred_x, curr_pred_y), (0, 0, 200))  # red - prediction
        cv2.imshow('image', img)


img = np.zeros((800, 800, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
