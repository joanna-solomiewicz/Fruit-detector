import cv2
import numpy as np


class ColorBasedImageSeparator:
    def __init__(self):
        self.contour_to_image_size_ratio = 0.01

    def color_separate_objects(self, image, color_ranges):
        assert isinstance(color_ranges, list)

        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        summary_mask = self._get_mask_for_ranges(hsv, color_ranges)

        summary_mask = cv2.erode(summary_mask, None, iterations=1)
        summary_mask = cv2.dilate(summary_mask, None, iterations=1)
        cv2.imshow('summary_filter', summary_mask)

        _, contours, hierarchy = cv2.findContours(summary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return self._get_outer_contours_of_enough_size(contours, hierarchy, image)

        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # after_first = np.zeros(gray.shape, np.uint8)
        # cv2.drawContours(after_first, outer_contours, -1, 255, -1)
        # cv2.imshow('after-first', after_first)
        #
        # gray = cv2.bitwise_and(gray, after_first)
        # cv2.imshow('after-fisrt-colors', gray)
        # canny = cv2.Canny(gray, self.threshold1, self.threshold2)
        # cv2.imshow('canny', canny)
        # _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # return contours

    def _get_mask_for_ranges(self, hsv_image, color_ranges):
        summary_mask = np.zeros(hsv_image.shape[0:2], np.uint8)
        for i, color_range in enumerate(color_ranges):
            mask = cv2.inRange(hsv_image, color_range[0], color_range[1])
            summary_mask = cv2.bitwise_or(summary_mask, mask)
        return summary_mask

    def _get_outer_contours_of_enough_size(self, contours, hierarchy, image):
        image_width, image_height = image.shape[:2]
        image_area = image_width * image_height
        outer_contours = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # only contours without parents (outer ones)
                if cv2.contourArea(contour) / image_area > self.contour_to_image_size_ratio:
                    outer_contours.append(contour)
        return outer_contours


class ImageSeparator:
    def __init__(self):
        self.threshold1 = 50
        self.threshold2 = 150

        self.contour_to_image_size_ratio = 0.01

    def canny_separate_objects(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        blur = cv2.GaussianBlur(blur, (3, 3), 0)
        cv2.imshow('blur', blur)
        # blur = cv2.bilateralFilter(gray, 11, 17, 17)
        # blur = cv2.medianBlur(grey, 3)
        canny = cv2.Canny(blur, self.threshold1, self.threshold2)

        # kernel = np.ones((3, 3), np.uint8)
        # close = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)

        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours

        # image_width, image_height = image.shape[:2]
        # image_area = image_width * image_height
        # contour_to_image_size_ratio = 0.01
        #
        # outer_contours = []
        # for i, contour in enumerate(contours):
        #     if hierarchy[0][i][3] == -1:  # only contours without parents (outer ones)
        #         if cv2.contourArea(contour) / image_area > contour_to_image_size_ratio:
        #             outer_contours.append(contour)
        # return outer_contours

    def saturation_separate_objects(self, image):
        # image = cv2.blur(image, (3, 3))
        # image = cv2.bilateralFilter(image, 11, 17, 17)
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        cv2.imshow('blur', blur)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # saturation = [element[1] for row in hsv for element in row]
        # grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hue = self.extract_channel(hsv, 0)
        saturation = self.extract_channel(hsv, 1)
        cv2.imshow('saturation', saturation)
        value = self.extract_channel(hsv, 2)
        # cv2.imshow('hue', hue)
        # cv2.imshow('saturation', saturation)
        # cv2.imshow('value', value)
        # _, thr = cv2.threshold(saturation, 80, 255, cv2.THRESH_BINARY)
        # thr = cv2.adaptiveThreshold(saturation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, thr = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('thr', thr)
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imshow('thr-close', thr)

        # canny = cv2.Canny(value, 80, 200)
        # cv2.imshow('hsv', hsv)
        # cv2.imshow('thr', thr)

        _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

        # image_width, image_height = image.shape[:2]
        # image_area = image_width * image_height
        #
        # outer_contours = []
        # for i, contour in enumerate(contours):
        #     if hierarchy[0][i][3] == -1:  # only contours without parents (outer ones)
        #         if cv2.contourArea(contour) / image_area > 0.2:
        #             outer_contours.append(contour)
        # return outer_contours

    def extract_channel(self, hsv, channel):
        return np.asarray([[element[channel] for element in row] for row in hsv])


class BlackBackgroundImageSeparator:
    def separate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image_width, image_height = image.shape[:2]
        image_area = image_width * image_height
        correct_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 0.1 * image_area:
                correct_contours.append(contour)
        if contours.__len__() > 0:
            return correct_contours[0]
        else:
            return None
