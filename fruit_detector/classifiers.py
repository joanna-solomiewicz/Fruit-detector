import cv2
import numpy as np


class Classifier:
    string_to_number_dictionary = {}
    number_to_string_dictionary = {}

    def classify(self, detected_features, database_features, database_fruit_names):
        correct_database_features_color, correct_database_features_hu, correct_database_fruit_names = self.convert_features_from_database(
            database_features, database_fruit_names)
        correct_features_color, correct_features_hu = self.convert_features_from_detector(detected_features)

        knn_hu = cv2.ml.KNearest_create()
        knn_hu.train(correct_database_features_hu, cv2.ml.ROW_SAMPLE, correct_database_fruit_names)
        ret_hu, results_hu, neighbours_hu, dist_hu = knn_hu.findNearest(correct_features_hu, 6)

        knn_color = cv2.ml.KNearest_create()
        knn_color.train(correct_database_features_color, cv2.ml.ROW_SAMPLE, correct_database_fruit_names)
        ret_color, results_color, neighbours_color, dist_color = knn_color.findNearest(correct_features_color, 6)

        fruit_names = []
        for i in range(detected_features.__len__()):
            neighbours = []
            for neighbour_hu in neighbours_hu[i]:
                neighbours.append(neighbour_hu)
            for neighbour_color in neighbours_color[i]:
                neighbours.append(neighbour_color)
            fruit_name = self.number_to_string_dictionary[self.most_common(neighbours)].split('_')[0]
            fruit_names.append(fruit_name)

        return fruit_names

    def most_common(self, lst):
        return max(set(lst), key=lst.count)

    def convert_feature_to_list(self, feature):
        correct_features_color = []
        correct_features_hu = []

        # colors
        for i in range(20):
            correct_features_color.append(feature.mean_color[0])  # Mean Hue
        correct_features_color.append(feature.mean_color[1])  # Mean saturation
        # correct_features_color.append(feature.mean_color[2]) # Mean Value - deleted to make better results
        correct_features_color.append(feature.standard_deviation[0])  # StdDev Heu
        correct_features_color.append(feature.standard_deviation[1])  # StdDev Saturation
        correct_features_color.append(feature.standard_deviation[2])  # StdDev Value

        # hu moments
        for i in range(7):
            correct_features_hu.append(feature.hu_moments[i] * 10000000000)  # multiplied to not loose precision

        return np.asarray(correct_features_color, dtype=np.float32), np.asarray(correct_features_hu, dtype=np.float32)

    def convert_features_from_detector(self, detected_features):
        correct_detected_features_color = []
        correct_detected_features_hu = []
        for detected_feature in detected_features:
            color, hu = self.convert_feature_to_list(detected_feature)
            correct_detected_features_color.append(color)
            correct_detected_features_hu.append(hu)
        return np.asarray(correct_detected_features_color, dtype=np.float32), np.asarray(correct_detected_features_hu,
                                                                                         dtype=np.float32)

    def convert_features_from_database(self, database_features, database_fruit_names):
        correct_database_features_color = []
        correct_database_features_hu = []
        for database_feature in database_features:
            color, hu = self.convert_feature_to_list(database_feature)
            correct_database_features_color.append(color)
            correct_database_features_hu.append(hu)

        correct_database_fruit_names = []
        self.make_dictionaries(database_fruit_names)
        for database_fruit_name in database_fruit_names:
            correct_database_fruit_names.append(self.string_to_number_dictionary[database_fruit_name])

        return np.asarray(correct_database_features_color, dtype=np.float32), np.asarray(correct_database_features_hu,
                                                                                         dtype=np.float32), np.asarray(
            correct_database_fruit_names, dtype=np.float32)

    def make_dictionaries(self, strings):
        for i, string in enumerate(set(strings)):
            self.string_to_number_dictionary[string] = i
            self.number_to_string_dictionary[i] = string
