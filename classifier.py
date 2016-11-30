class MinMaxEnum:
    min, max = range(2)


class ColorEnum:
    hue, saturation, value = range(3)


class Classifier:
    def __init__(self):
        self.name = ''
        self.roundness = []
        self.mean_hue = []

    def is_class(self, features):
        if self.mean_hue[0] <= features[0][0] <= self.mean_hue[1] and self.roundness[0] <= features[2] <= self.roundness[1]:
            return 1
        else:
            return 0

