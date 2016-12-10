import cv2
from fruit_detector.utils import get_mask, extract_channel
from fruit_detector.repositories import RangeRepository


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


def get_fruit_ranges(connection):
    range_repository = RangeRepository(connection)
    color_ranges = range_repository.find_all()[0]
    summary_ranges = _get_summary_ranges(color_ranges)
    return summary_ranges


def _get_summary_ranges(color_ranges):
    size = 180
    range_binary_list = [False] * size
    for color_range in color_ranges:
        for i in range(color_range[0], color_range[1]):
            range_binary_list[i] = True
    ranges = []
    in_range = False
    start = 0
    for i, v in enumerate(range_binary_list):
        if v is True and in_range is False:
            in_range = True
            start = i
        elif (v is False or i == size - 1) and in_range is True:
            ranges.append((start, i))
            in_range = False
    return ranges
