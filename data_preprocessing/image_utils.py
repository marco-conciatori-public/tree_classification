import numpy as np
import cv2
import random


def brightness(img, low: float, high: float):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def channel_shift(img, value: int):
    value = random.randint(-value, value)
    img = img + value
    img[:, :, :][img[:, :, :] > 255] = 255
    img[:, :, :][img[:, :, :] < 0] = 0
    img = img.astype(np.uint8)
    return img


def get_patch(img, size: int, top_left_coord: tuple):
    """
        Function to extract a patch form an image given pixel coordinates and size
    """
    patch = img[
            top_left_coord[1] : top_left_coord[1] + size,
            top_left_coord[0] : top_left_coord[0] + size,
    ].copy()
    return patch
