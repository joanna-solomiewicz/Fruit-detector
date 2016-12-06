import cv2
import argparse
import os
import sys
from separator.separator import BinaryImageSeparator
from detector.detector import FeatureDetector, RangeDetector
from repository.repository import FruitRepository, FeatureRepository, RangeRepository, init_database
import sqlite3

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

    for file_name in image_file_names:
        image = cv2.imread(directory_path + "/" + file_name)
        if image is None:
            continue
        fruit_name = file_name.split('.')[0]
        fruit_repository.add_if_not_exist(fruit_name)

        contour = separator.separate(image)

        feature = feature_detector.calculate_features(image, contour)
        feature_repository.add(feature, fruit_name)

        color_ranges = range_detector.get_color_ranges_in_contour(contour, image)
        for color_range in color_ranges:
            range_repository.add(color_range, fruit_name)

    #TODO delete later
    #     cv2.imshow(file_name, detector._get_mask(contour, image))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    connection.close()


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


def get_jpg_from_directory(directory_path):
    jpgFiles = []
    for file in os.listdir(directory_path):
        if file.endswith(".jpg") or file.endswith(".JPG"):
            jpgFiles.append(file)
    return jpgFiles


if __name__ == '__main__':
    main()
