from classifier import Classifier


class BananaClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.name = 'banana'
        self.mean_hue = [20, 35]
        self.roundness = [0, 0.5]

