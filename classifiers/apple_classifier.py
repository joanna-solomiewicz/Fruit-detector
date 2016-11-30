from classifier import Classifier


class AppleClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.name = 'apple'
        self.mean_hue = [0, 8]
        self.roundness = [0.5, 1]

