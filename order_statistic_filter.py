import cv2


def median_filter(image):
    median_image = image.copy()
    height = image.shape[0]
    width = image.shape[1]
    for row in range(height):
        for col in range(width):
            pixel = get_neighbors(row, col, image, 1)
            pixel.sort()
            median_image[row, col] = pixel[int(pixel.size/2)]

    return median_image


def max_filter(image):
    max_image = image.copy()
    height = image.shape[0]
    width = image.shape[1]
    for row in range(height):
        for col in range(width):
            pixel = get_neighbors(row, col, image, 1)
            max_image[row, col] = max(pixel)

    return max_image


def min_filter(image):
    median_image = image.copy()
    height = image.shape[0]
    width = image.shape[1]
    for row in range(height):
        for col in range(width):
            pixel = get_neighbors(row, col, image, 1)
            median_image[row, col] = min(pixel)

    return median_image


def midpoint_filter(image):
    midpoint_image = image.copy()
    height = image.shape[0]
    width = image.shape[1]
    for row in range(height):
        for col in range(width):
            pixel = get_neighbors(row, col, image, 1)
            midpoint_image[row, col] = min(pixel) / 2 + max(pixel) / 2

    return midpoint_image


def alpha_trimmed_mean(image):
    alpha_trimmed_image = image.copy()
    d = 4
    m = n = 3
    b = int((m * n) - d)
    trim_factor = int(d / 2)

    height = image.shape[0]
    width = image.shape[1]
    for row in range(height):
        for col in range(width):
            pixel = get_neighbors(row, col, image, 1)
            pixel.sort()
            pixel_size = pixel.size

            if b != 0:
                trimmed_pixel = pixel[trim_factor:pixel_size - trim_factor]
            else:
                trimmed_pixel = pixel

            pixel_sum = 0
            for pixel in trimmed_pixel:
                pixel_sum += pixel

            alpha_trimmed_image[row, col] = int(pixel_sum / b)

    return alpha_trimmed_image


def get_neighbors(row, col, img, distance):
    return img[max(row - distance, 0):min(row + distance + 1, img.shape[0]),
           max(col - distance, 0):min(col + distance + 1, img.shape[1])].flatten()


lenna = cv2.imread("Lenna.png", 0)
image = cv2.imread("uniform.png", 0)
restored_image = alpha_trimmed_mean(image)
cv2.imshow("lenna", lenna)
cv2.imshow("noise_image", image)
cv2.imshow("restored_image", restored_image)
cv2.waitKey(0)

# from matplotlib import pyplot as plt
#
# plt.figure("noise")
# plt.hist(image.ravel(), 256, [0, 256])
#
# plt.figure("restored")
# plt.hist(restored_image.ravel(), 256, [0, 256])
#
# plt.figure("lenna")
# plt.hist(lenna.ravel(), 256, [0, 256])
# plt.show()
