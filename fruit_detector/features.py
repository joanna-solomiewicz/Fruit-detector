import cv2
import numpy as np

from fruit_detector.utils import get_mask


class Feature:
    def __init__(self, mean_color, standard_deviation, hu_moments):
        self.mean_color = mean_color
        self.standard_deviation = standard_deviation
        self.hu_moments = hu_moments


class FeatureDetector:
    def calculate_features(self, image, contour):
        mask = get_mask(contour, image)
        mean_val_hsv, standard_deviation_hsv = self._calculate_mean_and_standard_deviation(image, mask)
        hu_moments = self._calculate_hu_moments(mask)
        return Feature(mean_val_hsv, standard_deviation_hsv, hu_moments)

    def _calculate_mean_and_standard_deviation(self, image, mask):
        mean_and_stdDev = cv2.meanStdDev(image, mask=mask)[:3]
        mean_val = np.uint8([x[0] for x in mean_and_stdDev[0]])  # in BGR
        mean_val_hsv = cv2.cvtColor(np.uint8([[mean_val]]), cv2.COLOR_BGR2HSV)[0][0]
        standard_deviation = np.uint8([x[0] for x in mean_and_stdDev[1]])  # in BGR
        standard_deviation_hsv = cv2.cvtColor(np.uint8([[standard_deviation]]), cv2.COLOR_BGR2HSV)[0][0]
        return mean_val_hsv, standard_deviation

    def _calculate_hu_moments(self, mask):
        hu_moments = cv2.HuMoments(cv2.moments(mask)).flatten()
        return hu_moments


