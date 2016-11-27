import cv2
from separator import ImageSeparator
from detector import FeatureDetector

from matplotlib import pyplot as plt
import skvideo.io
import skimage.io
import math
import numpy as np

image_name = 'i1.jpg'

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
    detector = FeatureDetector()
    image = cv2.imread(image_name)
    contours = separator.saturation_separate_objects(image)
    for i, contour in enumerate(contours):
        print(detector.calculate_features(image, contour, i))


    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    cv2.imshow('saturation-serparation', image)

    # image = cv2.imread(image_name)
    # objects = separator.canny_separate_objects(image)
    # cv2.drawContours(image, objects, -1, (0, 255, 0), 2)
    # cv2.imshow('canny-serparation', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
