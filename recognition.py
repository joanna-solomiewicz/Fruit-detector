import cv2
import argparse
import sqlite3
import sys
import random
from separator.separator import ColorBasedImageSeparator
from detector.detector import FeatureDetector
from classifiers.classifier import Classifier
from repository.repository import FeatureRepository, RangeRepository, FruitRepository

separator = ColorBasedImageSeparator()
detector = FeatureDetector()
classifier = Classifier()


def main():
    args = get_args()
    # image_path = get_image_path(args)
    image_path = 'img/inne/b2.jpg'
    db_path = get_db_path(args)

    image = cv2.imread(image_path)
    if image is None:
        print('Unable to open image.')
        sys.exit()

    connection = sqlite3.connect(db_path)
    feature_repository = FeatureRepository(connection)
    fruit_ranges_map = get_fruit_ranges_map(connection)

    contours = get_contours_of_fruits(fruit_ranges_map, image)
    if contours.__len__() > 0:
        detected_features = get_detected_features(contours, image)
        #TODO clean
        database_features, database_fruit_names = feature_repository.find_all()
        classified_contours_numbers, distances = classifier.classify(detected_features, database_features, database_fruit_names)
        classified_contours_names = []
        for i, classified_contours_number in enumerate(classified_contours_numbers):
            classified_contours_names.append(classifier.number_to_string_dictionary[classified_contours_number[0]])
            print('Found: ' + classified_contours_names[i])
            print(distances[i])
    else:
        print('No fruits found')
    #TODO delete later
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    connection.close()


def get_detected_features(contours, image):
    detected_features = []
    for contour in contours:
        detected_features.append(detector.calculate_features(image, contour))
    return detected_features


def get_contours_of_fruits(fruit_ranges_map, image):
    contours = []
    for fruit_name, fruit_range in fruit_ranges_map.items():
        objects = separator.color_separate_objects(image, fruit_range)
        cv2.drawContours(image, objects, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                         1)
        contours.extend(objects)
    return contours


def get_fruit_ranges_map(connection):
    fruit_repository = FruitRepository(connection)
    range_repository = RangeRepository(connection)
    fruit_names = fruit_repository.find_all()
    fruit_ranges_map = {}
    for fruit_name in fruit_names:
        color_ranges, _ = range_repository.find_by_fruit_name(fruit_name)
        summary_ranges = get_summary_ranges(color_ranges)
        fruit_ranges_map[fruit_name] = summary_ranges
    return fruit_ranges_map


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


if __name__ == '__main__':
    main()
