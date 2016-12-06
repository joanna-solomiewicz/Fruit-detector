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
    directory = 'img/recognition/'
    image_file_names = get_jpg_from_directory('img/recognition')
    db_path = get_db_path(args)
    connection = sqlite3.connect(db_path)

    range_repository = RangeRepository(connection)
    color_ranges, _ = range_repository.find_all()
    summary_ranges = get_summary_ranges(color_ranges)

    no_contours_found = 0
    good_guesses = 0
    bad_guesses = 0

    for image_file_name in image_file_names:
        fruit = image_file_name.split('.')[0].split('_')[0]
        image = cv2.imread(directory + image_file_name)
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
            classified_contours = classifier.classify(detected_features, database_features,
                                                      database_fruit_names)
            for classified_contour in classified_contours:
                print('Found: ' + classified_contour)
                if classified_contour != fruit:
                    print('Wrong, it was ' + fruit)
                    bad_guesses += 1
                else:
                    print('Good')
                    good_guesses += 1
                print('')
        else:
            print('No fruits found')
            no_contours_found += 1

        cv2.imshow('result', image)

        # cv2.waitKey(0)
        cv2.destroyAllWindows()

    connection.close()

    accuracy = 100 * good_guesses / (good_guesses + bad_guesses)
    print('Percent of good guesses: {}'.format(accuracy))
    print('Could not find contours in {} images out of {}.'.format(no_contours_found, image_file_names.__len__()))


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
