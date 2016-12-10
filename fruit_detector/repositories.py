from fruit_detector.features import Feature


class StoreException(Exception):
    def __init__(self, message, *errors):
        Exception.__init__(self, message)
        self.errors = errors


def init_database(connection):
    try:
        cursor = connection.cursor()
        table_name = 'fruits'
        cursor.execute(
            'DROP TABLE ' + table_name
        )
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS ' + table_name +
            '('
            'name VARCHAR(255) PRIMARY KEY'
            ' )'
        )
        print("Created table: " + table_name)

        table_name = 'features'
        cursor.execute(
            'DROP TABLE ' + table_name
        )
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS ' + table_name +
            '('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'fruit VARCHAR(255) REFERENCES fruits(name) NOT NULL,'
            'mean_color_h INTEGER NOT NULL,'
            'mean_color_s INTEGER NOT NULL,'
            'mean_color_v INTEGER NOT NULL,'
            'standard_deviation_h INTEGER NOT NULL,'
            'standard_deviation_s INTEGER NOT NULL,'
            'standard_deviation_v INTEGER NOT NULL,'
            'hu_1 DOUBLE NOT NULL,'
            'hu_2 DOUBLE NOT NULL,'
            'hu_3 DOUBLE NOT NULL,'
            'hu_4 DOUBLE NOT NULL,'
            'hu_5 DOUBLE NOT NULL,'
            'hu_6 DOUBLE NOT NULL,'
            'hu_7 DOUBLE NOT NULL'
            ')'
        )
        print("Created table: " + table_name)
        table_name = 'ranges'
        cursor.execute(
            'DROP TABLE ' + table_name
        )
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS ' + table_name +
            '('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'fruit VARCHAR(255) REFERENCES fruits(name) NOT NULL,'
            'min_hue INTEGER NOT NULL,'
            'max_hue INTEGER NOT NULL'
            ')'
        )
        print("Created table: " + table_name)

        connection.commit()
    except Exception:
        raise StoreException('Error while creating features table')


class FruitRepository:
    def __init__(self, connection):
        self._connection = connection

    def add_if_not_exist(self, fruit_name):
        if not self.exists(fruit_name):
            self.add(fruit_name)

    def add(self, fruit_name):
        try:
            cursor = self._connection.cursor()
            cursor.execute(
                'INSERT INTO fruits '
                '(name)'
                'VALUES(?)',
                (fruit_name,)
            )
            self._connection.commit()
        except Exception:
            raise StoreException('Error while adding feature')

    def exists(self, fruit_name):
        try:
            cursor = self._connection.cursor()
            cursor.execute(
                'SELECT * FROM fruits '
                'WHERE name = ?',
                (fruit_name,)
            )
            return not cursor.fetchone() is None
        except Exception:
            raise StoreException('Error while finding fruit ' + fruit_name)

    def find_all(self):
        try:
            cursor = self._connection.cursor()
            cursor.execute('SELECT * FROM fruits')
            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append(row[0])
            return result
        except Exception:
            raise StoreException('Error while finding all fruits')


class RangeRepository:
    def __init__(self, connection):
        self._connection = connection

    def add(self, color_range, fruit_name):
        try:
            cursor = self._connection.cursor()
            cursor.execute(
                'INSERT INTO ranges '
                '(fruit, min_hue, max_hue) '
                'VALUES(?,?,?)',
                (fruit_name, int(color_range[0]), int(color_range[1]))
            )
            self._connection.commit()
        except Exception:
            raise StoreException('Error while adding range')

    def find_by_fruit_name(self, fruit_name):
        try:
            cursor = self._connection.cursor()
            cursor.execute(
                'SELECT * FROM ranges '
                'WHERE fruit = ?',
                (fruit_name,)
            )
            rows = cursor.fetchall()
            color_ranges = []
            names = []
            for row in rows:
                fruit_name = row[1]
                color_range = (row[2], row[3])
                names.append(fruit_name)
                color_ranges.append(color_range)
            return color_ranges, names
        except Exception:
            raise StoreException('Error while finding ranges for fruit ' + fruit_name)

    def find_all(self):
        try:
            cursor = self._connection.cursor()
            cursor.execute('SELECT * FROM ranges')
            rows = cursor.fetchall()
            color_ranges = []
            names = []
            for row in rows:
                fruit_name = row[1]
                color_range = (row[2], row[3])
                names.append(fruit_name)
                color_ranges.append(color_range)
            return color_ranges, names
        except Exception:
            raise StoreException('Error while finding all ranges')


class FeatureRepository:
    def __init__(self, connection):
        self._connection = connection

    def add(self, feature, fruit_name):
        try:
            cursor = self._connection.cursor()
            cursor.execute(
                'INSERT INTO features '
                '(fruit, mean_color_h, mean_color_s, mean_color_v, '
                'standard_deviation_h, standard_deviation_s, standard_deviation_v,'
                'hu_1, hu_2, hu_3, hu_4, hu_5, hu_6, hu_7) '
                'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (fruit_name, int(feature.mean_color[0]), int(feature.mean_color[1]), int(feature.mean_color[2]),
                 int(feature.standard_deviation[0]), int(feature.standard_deviation[1]),
                 int(feature.standard_deviation[2]),
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

            features = []
            fruit_names = []
            for row in rows:
                mean_hsv = (row[2], row[3], row[4])
                standard_deviation_hsv = (row[5], row[6], row[7])
                hu_moments = row[8:15]
                features.append(Feature(mean_hsv, standard_deviation_hsv, hu_moments))
                fruit_names.append(row[1])

            return features, fruit_names
        except Exception:
            raise StoreException('Error while finding all features')
