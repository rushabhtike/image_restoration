import numpy as np
import cv2


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col = image.shape
        mean = 0
        var = 2
        sigma = var ** 4
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "reyleigh":
        row, col = image.shape
        mean_value = 15
        mode_value = np.sqrt(2 / np.pi) * mean_value
        reyleigh = np.random.rayleigh(mode_value, (row, col))
        reyleigh = reyleigh.reshape(row, col)
        noisy = image + reyleigh
        return noisy
    elif noise_typ == "uniform":
        row, col = image.shape
        uniform = np.random.uniform(-10, 30, (row, col))
        uniform = uniform.reshape(row, col)
        noisy = image + uniform
        return noisy
    elif noise_typ == "salt":
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255
        return out
    elif noise_typ == "pepper":
        # Pepper mode
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "sp":
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

image = cv2.imread("lenna.png", 0)
noisy_image = noisy("uniform", image)
noisy_image = noisy_image.astype('uint8')
cv2.imwrite('uniform.png', noisy_image)
