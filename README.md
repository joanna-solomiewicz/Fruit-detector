# Fruit detector #
Educational project associated with image processing and machine learning. Program must first be fed with prepared data. It uses k-NN algorithm for classification.

## Requirements ##
- gtk2
- opencv
- sklearn

## How to use ##
1. Clone this repository:
```
git clone https://github.com/joanna-solomiewicz/Fruit-detector.git
```
2. Feed application:
```
python feed.py --directory /path/to/directory/with/feed/data/
```
Images in this directory must be named in following convention:
```
FruitName[_Type].Number.jpg
```
e.g. Apple_green.1.jpg

3. Recognize fruits
```
python recognition.py --image /path/to/image/with/fruits/
```
Example results:
![zrzut ekranu w 2017-06-09 15-30-39](https://user-images.githubusercontent.com/11861292/26978797-063082b6-4d2d-11e7-99fa-40165f046d73.png)
