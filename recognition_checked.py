import cv2
import argparse
import sqlite3
import sys
import os
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
    directory = 'img/to_recognize/'
    image_file_names = get_jpg_from_directory('img/to_recognize/')
    db_path = get_db_path(args)
    connection = sqlite3.connect(db_path)

    range_repository = RangeRepository(connection)
    color_ranges, _ = range_repository.find_all()
    summary_ranges = get_summary_ranges(color_ranges)

    for image_file_name in image_file_names:
        image = cv2.imread(directory+image_file_name)
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
            classified_contours_numbers, distances = classifier.classify(detected_features, database_features,
                                                                         database_fruit_names)
            classified_contours_names = []
            for i, classified_contours_number in enumerate(classified_contours_numbers):
                classified_contours_names.append(classifier.number_to_string_dictionary[classified_contours_number[0]])
                print('Found: ' + classified_contours_names[i])
                if classified_contours_names[i] != image_file_name.split('.')[0]:
                    print('Wrong, it was '+image_file_name.split('.')[0])
                else:
                    print('Good')
                print(distances[i])
                print('')
        else:
            print('No fruits found')

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
        elif (v is False or i == size - 1) and in_range is True:
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


def get_jpg_from_directory(directory_path):
    jpgFiles = []
    for file in os.listdir(directory_path):
        if file.endswith(".jpg"):
            jpgFiles.append(file)
    return jpgFiles


if __name__ == '__main__':
    main()
