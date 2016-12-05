import cv2
import numpy as np


class Classifier:
    string_to_number_dictionary = {}
    number_to_string_dictionary = {}

    def classify(self, detected_features, database_features, database_fruit_names):
        correct_database_features, correct_database_fruit_names = self.convert_features_from_database(database_features, database_fruit_names)
        knn = cv2.ml.KNearest_create()
        knn.train(correct_database_features, cv2.ml.ROW_SAMPLE, correct_database_fruit_names)
        correct_features = self.convert_features_from_detector(detected_features)
        ret, results, neighbours, dist = knn.findNearest(correct_features, 3)
        return results

    def convert_feature_to_list(self, feature):
        correct_features = []
        for i in range(3):
            correct_features.append(feature.mean_color[i])
        for i in range(7):
            correct_features.append(feature.hu_moments[i])

        return np.asarray(correct_features, dtype=np.float32)

    def convert_features_from_detector(self, detected_features):
        correct_detected_features = []
        for detected_feature in detected_features:
            correct_detected_features.append(self.convert_feature_to_list(detected_feature))
        return np.asarray(correct_detected_features, dtype=np.float32)

    def convert_features_from_database(self, database_features, database_fruit_names):
        correct_database_features = []
        for database_feature in database_features:
            correct_database_features.append(self.convert_feature_to_list(database_feature))

        correct_database_fruit_names = []
        self.make_dictionaries(database_fruit_names)
        for database_fruit_name in database_fruit_names:
            correct_database_fruit_names.append(self.string_to_number_dictionary[database_fruit_name])

        return np.asarray(correct_database_features, dtype=np.float32), np.asarray(correct_database_fruit_names, dtype=np.float32)

    def make_dictionaries(self, strings):
        for i, string in enumerate(set(strings)):
            self.string_to_number_dictionary[string] = i
            self.number_to_string_dictionary[i] = string
