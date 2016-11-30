import cv2
from separator import ImageSeparator, ColorBasedImageSeparator
from detector import FeatureDetector
from classifier import Classifier
from classifier import MinMaxEnum
from classifier import ColorEnum

from matplotlib import pyplot as plt
import skvideo.io
import skimage.io
import math
import numpy as np

image_name = 'img/b1.jpg'

separator = ColorBasedImageSeparator()
detector = FeatureDetector()
classifier = Classifier()

fruit_ranges = {
    'red_apple': [
        ((0, 100, 50), (17, 255, 255)),
        ((168, 100, 50), (179, 255, 255)),
    ],
    'banana': [
        ((18, 100, 50), (35, 255, 255))
    ]
}

fruit_classifiers = {
    # 'red_apple': RedAppleClassifier,
    # 'banana': BananaClassifier
}


def main():
    image = cv2.imread(image_name)

    for fruit_name, color_range in fruit_ranges.items():
        contours = separator.color_separate_objects(image, color_range)
        for i, contour in enumerate(contours):
            features = detector.calculate_features(image, contour, i)
            # if fruit_classifiers.get(fruit_name).classify(features):
            #     draw(contours, fruit_name)
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)

    cv2.imshow('saturation-serparation', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()





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
