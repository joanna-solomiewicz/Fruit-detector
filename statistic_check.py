import cv2
import argparse
import sqlite3
import sys
from separator.separator import ColorBasedImageSeparator
from detector.detector import FeatureDetector
from classifiers.classifier import Classifier
from repository.repository import FeatureRepository, RangeRepository

from feed import get_jpg_from_directory
from recognition import get_fruit_ranges_map, get_contours_of_fruits, get_detected_features

separator = ColorBasedImageSeparator()
detector = FeatureDetector()
classifier = Classifier()


def main():
    args = get_args()
    # directory_path = get_directory_path(args)
    directory_path = 'img/recognition'
    image_file_names = get_jpg_from_directory(directory_path)
    db_path = get_db_path(args)

    connection = sqlite3.connect(db_path)
    range_repository = RangeRepository(connection)
    feature_repository = FeatureRepository(connection)

    fruit_ranges_map = get_fruit_ranges_map(connection)

    no_contours_found = 0
    good_guesses = 0
    bad_guesses = 0

    for image_file_name in image_file_names:
        image = cv2.imread(directory_path + image_file_name)
        if image is None:
            continue
        contours = get_contours_of_fruits(fruit_ranges_map, image)
        if contours.__len__() > 0:
            detected_features = get_detected_features(contours, image)

            # TODO clean
            database_features, database_fruit_names = feature_repository.find_all()
            classified_contours_numbers, distances = classifier.classify(detected_features, database_features,
                                                                         database_fruit_names)
            classified_contours_names = []
            for i, classified_contours_number in enumerate(classified_contours_numbers):
                classified_contours_names.append(classifier.number_to_string_dictionary[classified_contours_number[0]])
                print('Found: ' + classified_contours_names[i])
                if classified_contours_names[i] != image_file_name.split(' ')[0]:
                    print('Wrong, it was ' + image_file_name.split(' ')[0])
                    bad_guesses += 1
                else:
                    print('Good')
                    good_guesses += 1
                print(distances[i])
                print('')
        else:
            print('No fruits found')
            no_contours_found += 1

            # TODO delete_later
            # cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
            # cv2.imshow('result', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    connection.close()

    accuracy = 100 * good_guesses / (good_guesses + bad_guesses)
    print('Percent of good guesses: {}'.format(accuracy))
    print('Could not find contours in {} images out of {}.'.format(no_contours_found, image_file_names.__len__()))


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image")
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
