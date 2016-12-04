import cv2
import argparse
import sqlite3
import sys
from separator.separator import ColorBasedImageSeparator
from separator.separator import ImageSeparator
from detector.detector import FeatureDetector
from classifiers.apple_classifier import AppleClassifier
from classifiers.banana_classifier import BananaClassifier
import numpy as np

separator = ColorBasedImageSeparator()
detector = FeatureDetector()

fruit_ranges = {
    'red_apple': [
        ((0, 100, 50), (17, 255, 255)),
        ((168, 100, 50), (179, 255, 255)),
    ],
    'banana': [
        ((13, 80, 50), (27, 255, 255))
    ]
}


def main():
    args = get_args()
    # image_path = get_image_path(args)
    image_path = 'img/inne/b3.jpg'
    db_path = get_db_path(args)
    connection = sqlite3.connect(db_path)
    image = cv2.imread(image_path)
    if image is None:
        print('Unable to open image.')
        sys.exit()

    contours = separator.color_separate_objects(image, fruit_ranges.get('banana'))
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    # for fruit_name, color_range in fruit_ranges.items():
    #     contours = separator.color_separate_objects(image, color_range)
    #     for contour in contours:
    #         feature = detector.calculate_features(image, contour)
    #         if fruit_classifiers.get(fruit_name).is_class(feature):
    #             print('I found ' + fruit_name)
    #             cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)
    #             # draw(contours, fruit_name)

    cv2.imshow('result', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    connection.close()


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image")
    ap.add_argument("-db", "--data_base", help="path to sqlite database file")
    return vars(ap.parse_args())


def get_image_path(args):
    directoryPath = args.get("image", False)
    if not directoryPath:
        print('You must specify image using --image option.')
        sys.exit()
    return directoryPath


def get_db_path(args):
    db_path = args.get("data_base", False)
    if not db_path:
        return 'fruits.sqlite'
    return db_path


def extract_channel(hsv, channel):
    return np.asarray([[element[channel] for element in row] for row in hsv])


if __name__ == '__main__':
    main()
