from detector.feature import Feature
import sqlite3


class StoreException(Exception):
    def __init__(self, message, *errors):
        Exception.__init__(self, message)
        self.errors = errors


class FeatureRepository:
    def __init__(self, connection):
        self._connection = connection

    def create_table_if_not_exists(self):
        try:
            cursor = self._connection.cursor()
            table_name = 'features'
            cursor.execute(
                'CREATE TABLE IF NOT EXISTS ' + table_name +
                '('
                'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'fruit VARCHAR(255) NOT NULL,'
                'mean_color_h INTEGER NOT NULL,'
                'mean_color_s INTEGER NOT NULL,'
                'mean_color_v INTEGER NOT NULL,'
                'hu_1 DOUBLE NOT NULL,'
                'hu_2 DOUBLE NOT NULL,'
                'hu_3 DOUBLE NOT NULL,'
                'hu_4 DOUBLE NOT NULL,'
                'hu_5 DOUBLE NOT NULL,'
                'hu_6 DOUBLE NOT NULL,'
                'hu_7 DOUBLE NOT NULL'
                ')'
            )

            self._connection.commit()
        except Exception:
            raise StoreException('Error while creating features table')

    def add(self, feature, fruit_name):
        try:
            cursor = self._connection.cursor()
            cursor.execute(
                'INSERT INTO features '
                '(fruit, mean_color_h, mean_color_s, mean_color_v, hu_1, hu_2, hu_3, hu_4, hu_5, hu_6, hu_7) '
                'VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (fruit_name, feature.mean_color[0], feature.mean_color[1], feature.mean_color[2],
                 feature.hu_moments[0], feature.hu_moments[1], feature.hu_moments[2],
                 feature.hu_moments[3], feature.hu_moments[4], feature.hu_moments[5], feature.hu_moments[6])
            )
            self._connection.commit()
        except Exception:
            raise StoreException('Error while adding feature')

    def find_all(self):
        try:
            cursor = self._connection.cursor()
            cursor.execute('SELECT * FROM features')
            rows = cursor.fetchall()

            #TODO retun map fruit -> list of features
            features = []
            for row in rows:
                mean_hsv = (row[2], row[3], row[4])
                hu_moments = row[5:11]
                features.append(Feature(mean_hsv, hu_moments))

            return features

            self._connection.commit()
        except Exception:
            raise StoreException('Error while finding all features')
