import argparse
import sqlite3
import sys
from sklearn.metrics import confusion_matrix
import cv2

from fruit_detector.classifiers import Classifier
from fruit_detector.features import FeatureDetector
from fruit_detector.color_ranges import get_fruit_ranges
from fruit_detector.repositories import FeatureRepository, FruitRepository
from fruit_detector.separators import ColorBasedImageSeparator
from fruit_detector.utils import get_jpg_from_directory, find_fruit_on_image, get_base_fruit_name

detector = FeatureDetector()
classifier = Classifier()
separator = ColorBasedImageSeparator()


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
        detected_fruit_name_and_contour = find_fruit_on_image(image, fruit_color_ranges, feature_repository,
                                                              separator, classifier, detector)
        if detected_fruit_name_and_contour.__len__() == 0:
            print("  FAIL : Didn't detect any fruit.")
            no_objects_detected += 1
        elif detected_fruit_name_and_contour.__len__() == 1:
            detected_fruit = get_base_fruit_name(detected_fruit_name_and_contour[0][0])

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

    bad_guesses = singe_element_bad_guesses + no_objects_detected + multiple_objects_detected
    guesses = (good_guesses + bad_guesses)
    accuracy = 100 * good_guesses / guesses

    report_file = open('report.txt', 'w')
    matrix = confusion_matrix(detected, predicted)

    report_file.write('Percent of no objects detected: {}\n'.format(100 * no_objects_detected / guesses))
    report_file.write('Percent of multiple objects detected: {}\n'.format(100 * multiple_objects_detected / guesses))
    report_file.write('Percent of good guesses: {}\n'.format(accuracy))
    report_file.write('Percent of bad guesses: {}\n'.format(100 - accuracy))
    report_file.write(
        'Percent of bad guesses when single element was found: {}\n'.format(100 * singe_element_bad_guesses / guesses))

    report_confusion_matrix(matrix, number_to_string_dictionary, report_file)


def report_confusion_matrix(matrix, number_to_string_dictionary, report_file):
    report_file.write('\n\n')
    report_file.write('Confusion matrix:\n')
    separate_char = ';'
    report_file.write(separate_char)
    for fruit_name in number_to_string_dictionary.values():
        report_file.write(fruit_name + separate_char)
    report_file.write('\n')
    for i, row in enumerate(matrix):
        report_file.write(list(number_to_string_dictionary.values())[i] + separate_char)
        for element in row:
            report_file.write(str(element) + separate_char)
        report_file.write('\n')


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
