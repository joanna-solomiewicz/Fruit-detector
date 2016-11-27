class MinMaxEnum:
    min, max = range(2)


class ColorEnum:
    hue, saturation, value = range(3)


class Classifier:
    def __init__(self):
        self.fruit = []
        self.color = []

    def setColor(self, fruit, min, max):
        self.fruit.append(fruit)
        self.color.append([min, max])
