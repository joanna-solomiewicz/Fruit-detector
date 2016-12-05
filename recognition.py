import cv2
import argparse
import sqlite3
import sys
from separator.separator import ColorBasedImageSeparator
from detector.detector import FeatureDetector
from classifiers.classifier import Classifier
from repository.repository import FeatureRepository
from repository.repository import RangeRepository
import numpy as np

separator = ColorBasedImageSeparator()
detector = FeatureDetector()
classifier = Classifier()

def main():
    args = get_args()
    # image_path = get_image_path(args)
    image_path = 'img/inne/o4.jpg'
    db_path = get_db_path(args)
    connection = sqlite3.connect(db_path)

    range_repository = RangeRepository(connection)
    color_ranges, _ = range_repository.find_all()
    summary_ranges = get_summary_ranges(color_ranges)

    image = cv2.imread(image_path)
    feature_repository = FeatureRepository(connection)
    if image is None:
        print('Unable to open image.')
        sys.exit()

    contours = separator.color_separate_objects(image, summary_ranges)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    database_features, database_fruit_names = feature_repository.find_all()
    detected_features = []
    for contour in contours:
        detected_features.append(detector.calculate_features(image, contour))
    if contours.__len__() > 0:
        classified_contours_numbers, distances = classifier.classify(detected_features, database_features, database_fruit_names)
        classified_contours_names = []
        for i, classified_contours_number in enumerate(classified_contours_numbers):
            classified_contours_names.append(classifier.number_to_string_dictionary[classified_contours_number[0]])
            print('Found: ' + classified_contours_names[i])
            print(distances[i])
    else:
        print('No fruits found')

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


def get_summary_ranges(color_ranges):
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
        elif (v is False or i == size-1) and in_range is True:
            ranges.append((start, i))
            in_range = False
    return ranges


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
