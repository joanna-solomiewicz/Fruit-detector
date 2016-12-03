import cv2
import numpy as np
import math
from detector.feature import Feature


class FeatureDetector:
    def __init__(self):
        super().__init__()

    def calculate_features(self, image, contour):
        mask = self.get_mask(contour, image)
        mean_val_hsv = self.calculate_mean(image, mask)
        roundness = self.calculate_roundness(contour)
        return Feature(mean_val_hsv, roundness)

    def calculate_mean(self, image, mask):
        mean_val = np.uint8(cv2.mean(image, mask=mask)[:3])  # in BGR
        mean_val_hsv = cv2.cvtColor(np.uint8([[mean_val]]), cv2.COLOR_BGR2HSV)[0][0]
        return mean_val_hsv

    def get_mask(self, contour, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        return mask

    def extract_channel(self, hsv, channel):
        return np.asarray([[element[channel] for element in row] for row in hsv])

    def centre(contour):
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy

    def calculate_roundness(self, contour):
        area = cv2.contourArea(contour)
        if area > 0:
            _, enclosing_radius = cv2.minEnclosingCircle(contour)
            enclosing_area = math.pi * enclosing_radius**2
            return area/enclosing_area
        else:
            return -1