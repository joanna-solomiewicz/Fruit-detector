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
            cursor.execute('CREATE TABLE IF NOT EXISTS ' + table_name +
                           '('
                           'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                           'mean_color_h INTEGER NOT NULL,'
                           'mean_color_s INTEGER NOT NULL,'
                           'mean_color_v INTEGER NOT NULL,'
                           'roundness FLOAT (5) NOT NULL '
                           ')'
                           )

            self._connection.commit()
        except Exception:
            raise StoreException('Error while creating features table')

    def add(self, feature):
        try:
            cursor = self._connection.cursor()
            cursor.execute('INSERT INTO features (mean_color_h, mean_color_s, mean_color_v, roundness) VALUES(?,?,?,?)',
                           (feature.mean_color[0], feature.mean_color[1], feature.mean_color[2], feature.roundness)
                           )
            self._connection.commit()
        except Exception:
            raise StoreException('Error while adding feature')

    def find_all(self):
        try:
            cursor = self._connection.cursor()
            cursor.execute('SELECT * FROM features')
            rows = cursor.fetchall()

            for row in rows:
                print(row)

            self._connection.commit()
        except Exception:
            raise StoreException('Error while finding all features')
