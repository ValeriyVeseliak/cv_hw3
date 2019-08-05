import cv2 as cv
import numpy as np
import sys
import argparse


def get_patch_score(patch, template, method):
    if method == 'SAD':
        return np.sum(np.abs(patch - template))
    elif method == 'SSD':
        return np.sum((patch - template) ** 2)
    else:
        ssp = np.sum(patch ** 2)
        sst = np.sum(template ** 2)
        denominator = np.sqrt(ssp * sst)
        return np.sum(patch * template) / denominator


def match_template(image, template, method):
    if method != 'SAD' and method != 'SSD' and method != 'NCC':
        raise Exception("Unknown matching method")

    templ_width, templ_height, _ = template.shape
    image_width, image_height, _ = image.shape
    best_score = sys.maxsize if (method == 'SSD' or method == 'SAD') else 0
    best_patch_coords = ()
    for i in range(0, image_height - templ_height):
        for j in range(0, image_width - templ_width):
            start_point = (i, j)
            end_point = (i + templ_height, j + templ_width)
            patch = image[j:j + templ_width, i:i + templ_height]
            score = get_patch_score(patch, template, method)
            if ((method == 'SSD' or method == 'SAD') and score <= best_score) or \
                    (method == 'NCC' and score >= best_score):
                best_score = score
                best_patch_coords = (start_point, end_point)
    return best_patch_coords


parser = argparse.ArgumentParser()
parser.add_argument('--image', dest='image', type=str)
parser.add_argument('--template', dest='template', type=str)
parser.add_argument('--method', dest='method', type=str)
args = parser.parse_args()

img = cv.imread(args.image, cv.IMREAD_COLOR)
templ = cv.imread(args.template, cv.IMREAD_COLOR)

points = match_template(img, templ, args.method)
cv.rectangle(img, points[0], points[1], (0, 0, 0), 2, 8, 0)
cv.imshow("Result", img)
cv.waitKey(0)

