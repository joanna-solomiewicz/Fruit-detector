import cv2
from separator import ImageSeparator
from detector import FeatureDetector
from classifier import Classifier
from classifier import MinMaxEnum
from classifier import ColorEnum

from matplotlib import pyplot as plt
import skvideo.io
import skimage.io
import math
import numpy as np

image_name = '1.jpg'
image_ideal = '1.jpg'

blures = 10
threshold1 = 30
threshold2 = 200
minArea = 300.0
approximation = 10
separator = ImageSeparator()
detector = FeatureDetector()
classifier = Classifier()
minmax_enum = MinMaxEnum
color_enum = ColorEnum


def init_classifier():
    image = cv2.imread(image_ideal)
    contours = separator.saturation_separate_objects(image)
    for i, contour in enumerate(contours):
        mean_hsv = detector.calculate_features(image, contour, i)
        classifier.setColor('apple', mean_hsv, mean_hsv)


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

    init_classifier()

    image = cv2.imread(image_name)
    contours = separator.saturation_separate_objects(image)
    for i, contour in enumerate(contours):
        mean_hsv = detector.calculate_features(image, contour, i)
        fruit = classifier.fruit.index('apple')
        if classifier.color[fruit][minmax_enum.min][color_enum.hue] <= mean_hsv[color_enum.hue] <= classifier.color[fruit][minmax_enum.max][color_enum.hue]:
            print('Contour in classifier range of ' + classifier.fruit[fruit])

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
