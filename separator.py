import cv2
import numpy as np


class ImageSeparator:
    def __init__(self):
        self.threshold1 = 70
        self.threshold2 = 210

    def canny_separate_objects(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(grey, (3, 3))
        canny = cv2.Canny(blur, self.threshold1, self.threshold2)

        # kernel = np.ones((3, 3), np.uint8)
        # close = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)

        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def saturation_separate_objects(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # saturation = [element[1] for row in hsv for element in row]
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saturation = [[element[1] for element in row] for row in hsv]
        saturation = np.asarray(saturation)
        _, thr = cv2.threshold(saturation, 80, 200, cv2.THRESH_BINARY)
        cv2.imshow('hsv', hsv)
        cv2.imshow('thr', thr)

        _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
