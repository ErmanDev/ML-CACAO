import cv2
import numpy as np
from scipy.stats import skew

def auto_crop_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:  # Filter out tiny noise
            bean = img[y:y+h, x:x+w]
            crops.append(bean)

    return crops

def extract_hsv_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    def stats(channel):
        return [
            np.mean(channel),
            np.std(channel),
            np.var(channel),
            skew(channel.flatten())
        ]

    return stats(h) + stats(s) + stats(v)