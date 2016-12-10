import argparse
import sqlite3
import sys
import cv2

from fruit_detector.features import FeatureDetector, RangeDetector
from fruit_detector.repositories import FruitRepository, FeatureRepository, RangeRepository, init_database
from fruit_detector.separators import BinaryImageSeparator
from fruit_detector.utils import get_jpg_from_directory, get_base_fruit_name

separator = BinaryImageSeparator()
feature_detector = FeatureDetector()
range_detector = RangeDetector()


def main():
    args = get_args()
    directory_path = get_directory_path(args)
    db_path = get_db_path(args)
    image_file_names = get_jpg_from_directory(directory_path)

    connection = sqlite3.connect(db_path)
    init_database(connection)
    fruit_repository = FruitRepository(connection)
    feature_repository = FeatureRepository(connection)
    range_repository = RangeRepository(connection)

    added = 0

    for i, image_file_name in enumerate(image_file_names):
        print(str(i / image_file_names.__len__() * 100) + "%")

        image = cv2.imread(directory_path + image_file_name)
        if image is None:
            continue
        fruit_name = get_base_fruit_name(image_file_name)

        fruit_repository.add_if_not_exist(fruit_name)

        contour = separator.separate(image)

        feature = feature_detector.calculate_features(image, contour)
        feature_repository.add(feature, fruit_name)

        color_ranges = range_detector.get_color_ranges_in_contour(contour, image)
        for color_range in color_ranges:
            range_repository.add(color_range, fruit_name)

        added += 1
    connection.close()

    print("100%")
    print("FINISHED\n")
    print('Added ' + str(added) + ' fruits to database.')


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", help="path to the directory with images of fruits")
    ap.add_argument("-db", "--data_base", help="path to sqlite database file")
    return vars(ap.parse_args())


def get_directory_path(args):
    directoryPath = args.get("directory", False)
    if not directoryPath:
        print('You must specify images directory using --directory option.')
        sys.exit()
    return directoryPath


def get_db_path(args):
    db_path = args.get("data_base", False)
    if not db_path:
        return 'fruits.sqlite'
    return db_path


if __name__ == '__main__':
    main()
