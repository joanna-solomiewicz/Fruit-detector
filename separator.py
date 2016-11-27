import cv2
import numpy as np


class ImageSeparator:
    def __init__(self):
        self.threshold1 = 50
        self.threshold2 = 200

    def canny_separate_objects(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (3, 3))
        # blur = cv2.bilateralFilter(gray, 11, 17, 17)
        # blur = cv2.medianBlur(grey, 3)
        canny = cv2.Canny(blur, self.threshold1, self.threshold2)

        # kernel = np.ones((3, 3), np.uint8)
        # close = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)

        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def saturation_separate_objects(self, image):
        # image = cv2.blur(image, (3, 3))
        # image = cv2.bilateralFilter(image, 11, 17, 17)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # saturation = [element[1] for row in hsv for element in row]
        # grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hue = self.extract_channel(hsv, 0)
        saturation = self.extract_channel(hsv, 1)
        value = self.extract_channel(hsv, 2)
        # cv2.imshow('hue', hue)
        # cv2.imshow('saturation', saturation)
        # cv2.imshow('value', value)
        # _, thr = cv2.threshold(saturation, 80, 255, cv2.THRESH_BINARY)
        # thr = cv2.adaptiveThreshold(saturation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        blur = cv2.GaussianBlur(saturation, (5, 5), 0)
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # canny = cv2.Canny(value, 80, 200)
        # cv2.imshow('hsv', hsv)
        # cv2.imshow('thr', thr)

        _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def extract_channel(self, hsv, channel):
        return np.asarray([[element[channel] for element in row] for row in hsv])
