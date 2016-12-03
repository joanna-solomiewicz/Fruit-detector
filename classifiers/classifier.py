class MinMaxEnum:
    min, max = range(2)


class ColorEnum:
    hue, saturation, value = range(3)


class Classifier:
    def __init__(self):
        self.name = ''
        self.roundness = []
        self.mean_hue = []

    def is_class(self, feature):
        if self.mean_hue[0] <= feature.mean_color[0] <= self.mean_hue[1] \
                and self.roundness[0] <= feature.roundness <= self.roundness[1]:
            return 1
        else:
            return 0
