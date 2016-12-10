from sklearn import tree
import numpy as np


class Classifier:
    string_to_number_dictionary = {}
    number_to_string_dictionary = {}
    database_fruit_ids = []
    correct_database_features = []
    correct_detected_features = []
    classifier = None

    def __init__(self, database_features, database_fruit_names):
        self.make_dictionaries(database_fruit_names)
        self.convert_database(database_features, database_fruit_names)
        self.learn()

    def classify(self, detected_features):
        classified_fruits = self.predict(detected_features)

        return classified_fruits

    def learn(self):
        self.classifier = tree.DecisionTreeClassifier()
        self.classifier.fit(self.correct_database_features, self.database_fruit_ids)

    def predict(self, detected_features):
        self.convert_detected_features(detected_features)
        classified_fruits = []
        for correct_detected_feature in self.correct_detected_features:
            prediction = self.classifier.predict(np.reshape(correct_detected_feature, (1, -1)))
            classified_fruits.append(self.number_to_string_dictionary[prediction[0]])

        return classified_fruits

    def most_common(self, lst):
        return max(set(lst), key=lst.count)

    def make_dictionaries(self, strings):
        for i, string in enumerate(set(strings)):
            self.string_to_number_dictionary[string] = i
            self.number_to_string_dictionary[i] = string

    def string_to_id(self, strings):
        ids = []
        for string in strings:
            ids.append(self.string_to_number_dictionary[string])
        return ids

    def id_to_string(self, ids):
        strings = []
        for id in ids:
            strings.append(self.number_to_string_dictionary[id])
        return strings

    def convert_feature_to_list(self, feature):
        list = []
        for i in range(3):
            list.append(feature.mean_color[i])
        for i in range(7):
            list.append(feature.hu_moments[i])
        return list

    def convert_detected_features(self, detected_features):
        self.correct_detected_features = []
        for detected_feature in detected_features:
            self.correct_detected_features.append(self.convert_feature_to_list(detected_feature))

    def convert_database(self, database_features, database_fruit_names):
        self.database_fruit_ids = self.string_to_id(database_fruit_names)

        for database_feature in database_features:
            self.correct_database_features.append(self.convert_feature_to_list(database_feature))

