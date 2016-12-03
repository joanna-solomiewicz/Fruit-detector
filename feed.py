import cv2
import argparse
import os
import sys
from separator.separator import BlackBackgroundImageSeparator
from detector.detector import FeatureDetector
from detector.feature import Feature
from repository.repository import FeatureRepository
import sqlite3


def main():
    args = get_args()
    directory_path = get_directory_path(args)
    db_path = get_db_path(args)
    image_file_names = get_jpg_from_directory(directory_path)

    connection = sqlite3.connect(db_path)
    # featureRepository = FeatureRepository(connection)
    # featureRepository.create_table_if_not_exists()
    #
    # feautre1 = Feature((0, 0, 0), 0.5)
    # featureRepository.add(feautre1)

    separator = BlackBackgroundImageSeparator()
    detector = FeatureDetector()
    feature_repository = FeatureRepository(connection)

    for file_name in image_file_names:
        image = cv2.imread(file_name)
        contour = separator.separate(image)
        feature = detector.calculate_features(image, contour)
        feature_repository.add(feature)

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
        if file.endswith(".jpg"):
            jpgFiles.append(file)
    return jpgFiles


if __name__ == '__main__':
    main()
