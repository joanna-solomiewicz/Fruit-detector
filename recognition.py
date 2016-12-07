import cv2
import argparse
import sqlite3
import sys
import random
import os
from separator.separator import ColorBasedImageSeparator
from detector.detector import FeatureDetector
from classifiers.classifier import Classifier
from repository.repository import FeatureRepository, RangeRepository

separator = ColorBasedImageSeparator()
detector = FeatureDetector()
classifier = Classifier()


def main():
    args = get_args()
    directory_path = get_directory_path(args)
    if directory_path is None:
        image_path = get_image_path(args)
        image_paths = [image_path]
    else:
        image_paths = [directory_path + file_name for file_name in get_jpg_from_directory(directory_path)]

    db_path = get_db_path(args)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print('Unable to open image.')
            sys.exit()

        connection = sqlite3.connect(db_path)
        feature_repository = FeatureRepository(connection)
        fruit_color_ranges = get_fruit_ranges(connection)

        detected_fruits = find_fruit_on_image(image, fruit_color_ranges, feature_repository)
        for fruit_name, contour in detected_fruits:
            print_name_in_center(contour, fruit_name, image)

        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    connection.close()


def find_fruit_on_image(image, fruit_color_ranges, feature_repository):
    detected_contours = separator.color_separate_objects(image, fruit_color_ranges)
    if detected_contours.__len__() == 0:
        return []
    detected_features = get_detected_features(detected_contours, image)
    database_features, database_fruit_names = feature_repository.find_all()
    classified_fruit_names = classifier.classify(detected_features, database_features, database_fruit_names)
    # for i, cont in enumerate(detected_contours):
    #     print_name_in_center(cont, classified_fruit_names[i], image)
    # cv2.imshow('res', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return list(zip(classified_fruit_names, detected_contours))


def get_fruits_matching_color_range(classified_fruit_names, detected_contours, detected_fruit_names):
    correctly_classified = []
    for i, fruit_name in enumerate(classified_fruit_names):
        detected_fruit = detected_fruit_names[i].split('_')[0]
        if fruit_name == detected_fruit:
            correctly_classified.append((fruit_name, detected_contours[i]))
    return correctly_classified


def print_name_in_center(contour, name, image):
    centre = get_centre(contour)
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1
    fontThickness = 1
    fontSize = cv2.getTextSize(name, fontFace, fontScale, fontThickness)[0]
    fontOrg = (int(centre[0] - fontSize[0] / 2), int(centre[1] - fontSize[1] / 2))

    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    cv2.putText(image, name, fontOrg, fontFace, fontScale, color, fontThickness)
    cv2.drawContours(image, [contour], -1, color, 1)


def get_centre(contour):
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def get_detected_features(contours, image):
    detected_features = []
    for contour in contours:
        detected_features.append(detector.calculate_features(image, contour))
    return detected_features


def get_fruit_ranges(connection):
    range_repository = RangeRepository(connection)
    color_ranges = range_repository.find_all()[0]
    summary_ranges = get_summary_ranges(color_ranges)
    return summary_ranges


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


def get_jpg_from_directory(directory_path):
    jpgFiles = []
    for file in os.listdir(directory_path):
        if file.endswith(".jpg") or file.endswith(".JPG"):
            jpgFiles.append(file)
    return jpgFiles


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image")
    ap.add_argument("-d", "--directory", help="path to the directory with images of fruits")
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


def get_directory_path(args):
    directoryPath = args.get("directory", False)
    if not directoryPath:
        return None
    return directoryPath


if __name__ == '__main__':
    main()
