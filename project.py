import cv2
from matplotlib import pyplot as plt
import skvideo.io
import skimage.io
import math
import numpy as np

image_name = 'ball.jpg'

blures = 10
threshold1 = 30
threshold2 = 200

def main():
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    balls = find_balls(img)
    img2 = cv2.imread(image_name, 1)

    for ball in balls:
        draw_ball(img2, ball)

    skimage.io.imshow(img2)
    skimage.io.show()


def draw_ball(img2, ball):
    cv2.circle(img2, (int(ball[0]), int(ball[1])), int(ball[2]), (255, 0, 0), 2)
    cv2.circle(img2, (int(ball[0]), int(ball[1])), 2, (0, 0, 255), -1)


def find_balls(img):
    for i in range(blures):
        img = cv2.medianBlur(img, 5)

    skimage.io.imshow(img)
    skimage.io.show()
    img = cv2.Canny(img, threshold1, threshold2)

    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.dilate(img,kernel,iterations = 3)
    # cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    skimage.io.imshow(img)
    skimage.io.show()

    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img2 = cv2.imread(image_name, 1)
    cv2.drawContours(img2, contours, -1, (0, 255, 0), 3)

    skimage.io.imshow(img2)
    skimage.io.show()

    balls = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 300.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        r = math.sqrt(area / math.pi)

        approximation = 10
        if w - approximation <= h <= w + approximation:
            d = (w + h) / 2
            if d - approximation <= 2 * r <= d + approximation:
                x, y = centre(contour)
                balls.append((x, y, r))
    return balls


def centre(contour):
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy

    # cap = cv2.VideoCapture('test2.mp4')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
