import cv2
import numpy as np
import colorsys


class FeatureDetector:
    def __init__(self):
        super().__init__()

    def calculate_features(self, image, contour, id):
        mask = self.get_mask(contour, image, id)
        mean_val_hsv = self.calculate_mean(image, mask)

        return mean_val_hsv

    def calculate_mean(self, image, mask):
        mean_val = np.uint8(cv2.mean(image, mask=mask)[:3])  # in BGR
        mean_val_hsv = cv2.cvtColor(np.uint8([[mean_val]]), cv2.COLOR_BGR2HSV)[0][0]
        return mean_val_hsv

    def get_mask(self, contour, image, id):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        cv2.imshow('contour' + str(id), mask)
        return mask

    def extract_channel(self, hsv, channel):
        return np.asarray([[element[channel] for element in row] for row in hsv])

    def centre(contour):
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy
