import cv2
import numpy as np
import math
from detector.feature import Feature


def extract_channel(hsv, channel):
    return np.asarray([[element[channel] for element in row] for row in hsv])


def get_mask(contour, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


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


class RangeDetector:
    def get_color_ranges_in_contour(self, contour, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = extract_channel(hsv, 0)
        mask = get_mask(contour, image)
        hist = cv2.calcHist([hue], [0], mask, [180], [0, 179])
        all_pixels_number = sum(hist)[0]
        self._filter_hist(hist, 0.001, all_pixels_number)
        ranges = self._get_ranges(hist)
        self._filter_ranges(ranges, hist, 0.01, all_pixels_number)
        return ranges

    def _filter_hist(self, hist, factor, all_pixels_number):
        for i, v in enumerate(hist):
            if v[0] < factor * all_pixels_number:
                hist[i][0] = 0

    def _get_ranges(self, hist):
        ranges = []
        in_range = False
        start = 0
        for i, v in enumerate(hist):
            if v > 0 and in_range is False:
                in_range = True
                start = i
            elif (v == 0 or i == hist.size - 1) and in_range is True:
                ranges.append((start, i))
                in_range = False
        return ranges

    def _filter_ranges(self, ranges, hist, factor, all_pixels_number):
        for r in ranges[:]:
            if r[1] - r[0] < 5:
                range_pixels = 0
                for i in range(r[0], r[1] + 1):
                    range_pixels += hist[i][0]
                if range_pixels < factor * all_pixels_number:
                    ranges.remove(r)
