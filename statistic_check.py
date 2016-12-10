import cv2
import argparse
import sqlite3
import sys
from separator.separator import ColorBasedImageSeparator
from detector.detector import FeatureDetector
from classifiers.classifier import Classifier
from repository.repository import FeatureRepository, FruitRepository
from sklearn.metrics import confusion_matrix

from feed import get_jpg_from_directory
from recognition import find_fruit_on_image, get_fruit_ranges

separator = ColorBasedImageSeparator()
detector = FeatureDetector()
classifier = Classifier()


def main():
    args = get_args()
    directory_path = get_directory_path(args)
    image_file_names = get_jpg_from_directory(directory_path)
    db_path = get_db_path(args)

    connection = sqlite3.connect(db_path)
    feature_repository = FeatureRepository(connection)
    fruit_color_ranges = get_fruit_ranges(connection)

    fruit_repository = FruitRepository(connection)
    fruit_names = fruit_repository.find_all()

    string_to_number_dictionary = {}
    number_to_string_dictionary = {}
    for i, fruit_name in enumerate(set(fruit_names)):
        fruit_name = fruit_name.split('_')[0]
        string_to_number_dictionary[fruit_name] = i
        number_to_string_dictionary[i] = fruit_name

    good_guesses = 0
    singe_element_bad_guesses = 0
    no_objects_detected = 0
    multiple_objects_detected = 0

    predicted = []
    detected = []

    for image_file_name in image_file_names:
        print("Image:" + image_file_name)
        fruit_on_image = image_file_name.split('.')[0].split('_')[0]
        image = cv2.imread(directory_path + image_file_name)
        if image is None:
            print("  WARN : Unable to open file.")
            continue
        detected_fruit_name_and_contour = find_fruit_on_image(image, fruit_color_ranges, feature_repository)
        if detected_fruit_name_and_contour.__len__() == 0:
            print("  FAIL : Didn't detect any fruit.")
            no_objects_detected += 1
        elif detected_fruit_name_and_contour.__len__() == 1:
            detected_fruit = detected_fruit_name_and_contour[0][0].split('_')[0]

            predicted.append(string_to_number_dictionary[fruit_on_image])
            detected.append(string_to_number_dictionary[detected_fruit])

            if detected_fruit == fruit_on_image:
                print("  SUCCESS : Detected " + detected_fruit + ".")
                good_guesses += 1
            else:
                print("  FAIL : Detected " + detected_fruit + ", but it was " + fruit_on_image + ".")
                singe_element_bad_guesses += 1
        else:
            detected_fruit_names = [i[0] for i in detected_fruit_name_and_contour]
            print("  FAIL : Detected too many fruits: " + ', '.join(
                detected_fruit_names) + ", but it was " + fruit_on_image + ".")
            multiple_objects_detected += 1

    connection.close()

    bad_guesses = singe_element_bad_guesses + no_objects_detected + multiple_objects_detected;
    guesses = (good_guesses + bad_guesses)
    accuracy = 100 * good_guesses / guesses
    print('Percent of no objects detected: {}'.format(100 * no_objects_detected / guesses))
    print('Percent of multiple objects detected: {}'.format(100 * multiple_objects_detected / guesses))
    print()
    print('Percent of bad guesses when single element was found: {}'.format(100 * singe_element_bad_guesses / guesses))
    print('Percent of good guesses: {}'.format(accuracy))
    print('Percent of bad guesses: {}'.format(100 - accuracy))

    matrix = confusion_matrix(detected, predicted)
    print('\nConfusion matrix: ')
    print(matrix)


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
