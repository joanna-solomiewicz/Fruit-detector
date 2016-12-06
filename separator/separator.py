import cv2
import numpy as np


def get_image_area(image):
    image_width, image_height = image.shape[:2]
    image_area = image_width * image_height
    return image_area


class ColorBasedImageSeparator:
    def color_separate_objects(self, image, hue_ranges):
        assert isinstance(hue_ranges, list)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        color_ranges = self._generate_color_ranges(hue_ranges, 80, 80)
        summary_mask = self._get_mask_from_ranges(hsv, color_ranges)
        summary_mask = cv2.morphologyEx(summary_mask, cv2.MORPH_OPEN, None, iterations=2)
        _, contours, hierarchy = cv2.findContours(summary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        outer_contours = self._get_outer_contours_of_enough_size(contours, hierarchy, 0.01, image)
        return outer_contours

    def _generate_color_ranges(self, hue_ranges, min_saturation=80, min_value=80):
        color_ranges = []
        for hue_range in hue_ranges:
            color_ranges.append(((hue_range[0], min_saturation, min_value), (hue_range[1], 255, 255)))
        return color_ranges

    def _get_mask_from_ranges(self, hsv_image, color_ranges):
        summary_mask = np.zeros(hsv_image.shape[0:2], np.uint8)
        for i, color_range in enumerate(color_ranges):
            mask = cv2.inRange(hsv_image, color_range[0], color_range[1])
            summary_mask = cv2.bitwise_or(summary_mask, mask)
        return summary_mask

    def _get_outer_contours_of_enough_size(self, contours, hierarchy, factor, image):
        image_area = get_image_area(image)
        outer_contours = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # only contours without parents (outer ones)
                if cv2.contourArea(contour) / image_area > factor:
                    outer_contours.append(contour)
        return outer_contours


class BinaryImageSeparator:
    def separate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_area = get_image_area(image)
        correct_contours = []
        self._get_contours_of_enougt_size(contours, correct_contours, image_area)
        if contours.__len__() > 0:
            return correct_contours[0]
        else:
            return None

    def _get_contours_of_enougt_size(self, contours, correct_contours, image_area):
        for contour in contours:
            if cv2.contourArea(contour) > 0.1 * image_area:
                correct_contours.append(contour)
