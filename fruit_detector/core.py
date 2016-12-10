def find_fruit_on_image(image, fruit_color_ranges, feature_repository, separator, classifier, detector):
    detected_contours = separator.color_separate_objects(image, fruit_color_ranges)
    if detected_contours.__len__() == 0:
        return []
    detected_features = _get_detected_features(detector, detected_contours, image)
    database_features, database_fruit_names = feature_repository.find_all()
    classified_fruit_names = classifier.classify(detected_features, database_features, database_fruit_names)
    # for i, cont in enumerate(detected_contours):
    #     print_name_in_center(cont, classified_fruit_names[i], image)
    # cv2.imshow('res', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return list(zip(classified_fruit_names, detected_contours))


def _get_fruits_matching_color_range(classified_fruit_names, detected_contours, detected_fruit_names):
    correctly_classified = []
    for i, fruit_name in enumerate(classified_fruit_names):
        detected_fruit = detected_fruit_names[i].split('_')[0]
        if fruit_name == detected_fruit:
            correctly_classified.append((fruit_name, detected_contours[i]))
    return correctly_classified


def _get_detected_features(detector, contours, image):
    detected_features = []
    for contour in contours:
        detected_features.append(detector.calculate_features(image, contour))
    return detected_features
