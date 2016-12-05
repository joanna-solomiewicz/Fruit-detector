import cv2
import argparse
import os
import sys
from separator.separator import BinaryImageSeparator
from detector.detector import FeatureDetector
from repository.repository import FeatureRepository
from classifiers.classifier import Classifier
import sqlite3


def main():
    args = get_args()
    directory_path = get_directory_path(args)
    db_path = get_db_path(args)
    image_file_names = get_jpg_from_directory(directory_path)
    connection = sqlite3.connect(db_path)

    separator = BinaryImageSeparator()
    detector = FeatureDetector()
    feature_repository = FeatureRepository(connection)
    classifier = Classifier()

    feature_repository.create_table_if_not_exists()
    database_features, database_fruit_names = feature_repository.find_all()

    for file_name in image_file_names:
        image = cv2.imread(directory_path + "/" + file_name)
        if image is None:
            continue

        contour = separator.separate(image)
        # delete later
        # cv2.imshow(file_name, detector._get_mask(contour, image))
        detected_feature = detector.calculate_features(image, contour)
        color_ranges = detector.get_color_ranges_in_contour(contour, image)
        print(color_ranges)
        classifier.classify(detected_feature, database_features, database_fruit_names)
        fruit_name = file_name.split('.')[0]
        feature_repository.add(detected_feature, fruit_name)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # features = feature_repository.find_all()
    # for f in features:
    #     print(f.mean_color)
    #     print(f.hu_moments)

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
