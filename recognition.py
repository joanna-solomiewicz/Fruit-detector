import argparse
import sqlite3
import sys
import cv2

from fruit_detector.core import find_fruit_on_image
from fruit_detector.features import FeatureDetector
from fruit_detector.color_ranges import get_fruit_ranges
from fruit_detector.repositories import FeatureRepository
from fruit_detector.classifiers import Classifier
from fruit_detector.separators import ColorBasedImageSeparator
from fruit_detector.utils import get_jpg_from_directory, print_name_in_center

separator = ColorBasedImageSeparator()
detector = FeatureDetector()
classifier = Classifier()


def main():
    args = get_args()
    directory_path = get_directory_path(args)
    db_path = get_db_path(args)
    images_paths = get_images_paths(args, directory_path)

    connection = sqlite3.connect(db_path)
    feature_repository = FeatureRepository(connection)

    fruit_color_ranges = get_fruit_ranges(connection)

    for image_path in images_paths:
        image = cv2.imread(image_path)
        if image is None:
            print('Unable to open image.')
            sys.exit()

        detected_fruits = find_fruit_on_image(image, fruit_color_ranges, feature_repository, separator, classifier,
                                              detector)
        for fruit_name, contour in detected_fruits:
            print_name_in_center(contour, fruit_name, image)
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    connection.close()


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image")
    ap.add_argument("-d", "--directory", help="path to the directory with images of fruits")
    ap.add_argument("-db", "--data_base", help="path to sqlite database file")
    return vars(ap.parse_args())


def get_directory_path(args):
    directoryPath = args.get("directory", False)
    if not directoryPath:
        return None
    return directoryPath


def get_db_path(args):
    db_path = args.get("data_base", False)
    if not db_path:
        return 'fruits.sqlite'
    return db_path


def get_images_paths(args, directory_path):
    if directory_path is None:
        image_path = get_image_path(args)
        image_paths = [image_path]
    else:
        image_paths = [directory_path + file_name for file_name in get_jpg_from_directory(directory_path)]
    return image_paths


def get_image_path(args):
    directoryPath = args.get("image", False)
    if not directoryPath:
        print('You must specify image using --image option.')
        sys.exit()
    return directoryPath


if __name__ == '__main__':
    main()
