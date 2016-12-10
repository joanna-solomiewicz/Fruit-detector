import os
import cv2
import numpy as np


def get_jpg_from_directory(directory_path):
    jpgFiles = []
    for file in os.listdir(directory_path):
        if file.endswith(".jpg") or file.endswith(".JPG"):
            jpgFiles.append(file)
    return jpgFiles


def get_base_fruit_name(string):
    return string.split('.')[0].split('_')[0]


def extract_channel(hsv, channel):
    return np.asarray([[element[channel] for element in row] for row in hsv])


def get_mask(contour, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def find_fruit_on_image(image, fruit_color_ranges, feature_repository, separator, classifier, detector):
    detected_contours = separator.color_separate_objects(image, fruit_color_ranges)
    if detected_contours.__len__() == 0:
        return []
    detected_features = get_detected_features(detector, detected_contours, image)
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
    color = (0, 0, 0)
    cv2.putText(image, name, fontOrg, fontFace, fontScale, color, fontThickness)


def get_centre(contour):
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def get_detected_features(detector, contours, image):
    detected_features = []
    for contour in contours:
        detected_features.append(detector.calculate_features(image, contour))
    return detected_features

