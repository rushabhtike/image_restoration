import cv2
import numpy as np
import math


def arithmetic_filter(image):
    val = 0
    pixels = 0
    height = image.shape[0]
    width = image.shape[1]
    arithmetic_image = image.copy()
    for row in range(height):
        for col in range(width):
            pixel = get_neighbors(row, col, image, 1)
            total = sum(pixel)
            pixels = total / 9
            arithmetic_image[row][col] = pixels
    return arithmetic_image

def geometric_filter(image):
    height = image.shape[0]
    width = image.shape[1]
    val = 0
    total1 = 0
    # product = 1
    geo_image = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            pixel = get_neighbors(row, col, image, 1)
            prod = 1
            m = 1/len(pixel)
            for p in pixel:
                p = math.pow(p, m)
                prod *= p
            geo_image[row][col] = int(prod)
    geo_image.astype(int)
    return geo_image

