import cv2
from separator import ImageSeparator

from matplotlib import pyplot as plt
import skvideo.io
import skimage.io
import math
import numpy as np

image_name = '1.jpg'

blures = 10
threshold1 = 30
threshold2 = 200
minArea = 300.0
approximation = 10


def main():
    # video_stream = cv2.VideoCapture(0)
    #
    # while video_stream.isOpened():
    #     is_success, frame = video_stream.read()
    #     if not is_success:
    #         break
    #
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # video_stream.release()

    separator = ImageSeparator()
    image = cv2.imread(image_name)

    objects = separator.saturation_separate_objects(image)

    cv2.drawContours(image, objects, -1, (0, 255, 0), 3)

    cv2.imshow('image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def draw_ball(img, ball):
#     cv2.circle(img, (int(ball[0]), int(ball[1])), int(ball[2]), (255, 0, 0), 2)
#     cv2.circle(img, (int(ball[0]), int(ball[1])), 2, (0, 0, 255), -1)


# def find_balls(img):
#     balls = []
#
#     for i in range(blures):
#         img = cv2.medianBlur(img, 5)
#
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     cv2.imshow('blur', img)
#
#     # boundries = [(110, 130)]
#     boundries = [(15, 45), (45, 75), (75, 105), (105, 135), (135, 165), (165, 195), (195, 225)]
#
#     for lower, upper in boundries:
#         lower_bound = (lower, 50, 50)
#         upper_bound = (upper, 255, 255)
#
#         mask = cv2.inRange(hsv, lower_bound, upper_bound)
#         # mask = cv2.erode(mask, None, iterations=2)
#         # mask = cv2.dilate(mask, None, iterations=2)
#
#
#         cv2.imshow('mask', mask)
#
#         img = cv2.Canny(mask, threshold1, threshold2)
#
#         # img = cv2.dilate(img,kernel,iterations = 3)
#         # kernel = np.ones((5, 5), np.uint8)
#         # cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#
#         # cv2.imshow('canny', img)
#
#         img2 = img[:]
#
#         img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#         # cv2.drawContours(img2, contours, -1, (0, 255, 0), 3)
#
#         # cv2.imshow('edges', img2)
#
#
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area <= minArea:
#                 continue
#             _, _, w, h = cv2.boundingRect(contour)
#             r = math.sqrt(area / math.pi)
#
#             if w - approximation <= h <= w + approximation:
#                 d = (w + h) / 2
#                 if d - approximation <= 2 * r <= d + approximation:
#                     x, y = centre(contour)
#                     balls.append((x, y, r, area))
#
#     return balls


def centre(contour):
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy


if __name__ == '__main__':
    main()
