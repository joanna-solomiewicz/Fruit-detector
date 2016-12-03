import cv2
from separator import ColorBasedImageSeparator
from detector import FeatureDetector
from classifiers.apple_classifier import AppleClassifier
from classifiers.banana_classifier import BananaClassifier

images = ['img/czj2.jpg', 'img/b1.jpg']

separator = ColorBasedImageSeparator()
detector = FeatureDetector()

fruit_ranges = {
    'red_apple': [
        ((0, 100, 50), (17, 255, 255)),
        ((168, 100, 50), (179, 255, 255)),
    ],
    'banana': [
        ((18, 100, 50), (35, 255, 255))
    ]
}

fruit_classifiers = {
    'red_apple': AppleClassifier(),
    'banana': BananaClassifier()
}


def main():
    for file in images:
        image = cv2.imread(file)

        for fruit_name, color_range in fruit_ranges.items():
            contours = separator.color_separate_objects(image, color_range)
            for i, contour in enumerate(contours):
                feature = detector.calculate_features(image, contour, i)
                if fruit_classifiers.get(fruit_name).is_class(feature):
                    print('I found ' + fruit_name)
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)
                    # draw(contours, fruit_name)

        cv2.imshow('saturation-serparation', image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

