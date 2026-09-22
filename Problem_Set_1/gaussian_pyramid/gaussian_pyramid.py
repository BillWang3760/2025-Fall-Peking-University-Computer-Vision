import numpy as np
from PIL import Image


def cross_correlation_2d(input_image_array, filter):
    w = input_image_array.shape[0]
    h = input_image_array.shape[1]
    m = filter.shape[0]
    n = filter.shape[1]
    p = m // 2
    q = n // 2
    output_image_array = np.zeros_like(input_image_array, dtype='float64')
    for i in range(w):
        for j in range(h):
            for k in range(-p, p + 1):
                for l in range(-q, q + 1):
                    if 0 <= i + k < w and 0 <= j + l < h:
                        output_image_array[i][j] += input_image_array[i + k][j + l] * filter[k][l]
    return output_image_array


def convolve_2d(input_image_array, filter):
    return cross_correlation_2d(input_image_array, np.transpose(filter))


def gaussian_blur_kernel_2d(kernel_size, standard_deviation):
    p = kernel_size // 2
    gaussian_kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussian_kernel[i][j] = np.exp(-((i - p) * (i - p) + (j - p) * (j - p)) / (2 * standard_deviation * standard_deviation))
    gaussian_kernel /= np.sum(gaussian_kernel)
    return gaussian_kernel


def low_pass(input_image_array, kernel_size, standard_deviation):
    filter = gaussian_blur_kernel_2d(kernel_size, standard_deviation)
    out_image_array = convolve_2d(input_image_array, filter)
    return out_image_array


def image_subsampling(input_image_array):
    w = input_image_array.shape[0]
    h = input_image_array.shape[1]
    c = input_image_array.shape[2]
    w_tilda = w // 2
    h_tilda = h // 2
    output_image_array = np.zeros((w_tilda, h_tilda, c))
    for i in range(w_tilda):
        for j in range(h_tilda):
            output_image_array[i][j] = input_image_array[2 * i][2 * j]
    return output_image_array


def gaussian_pyramid(input_image_array, pyramid_level):
    pyramid = [input_image_array]
    for i in range(pyramid_level - 1):
        blur_image_array = low_pass(input_image_array, 3, 1)
        subsample_image_array = image_subsampling(blur_image_array)
        pyramid.append(subsample_image_array)
        input_image_array = subsample_image_array
    return pyramid


def output_gaussian_pyramid(file_name, pyramid_level):
    input_image = Image.open(f'{file_name}')
    input_image_array = np.array(input_image)
    pyramid = gaussian_pyramid(input_image_array, pyramid_level)
    file_name, extension = file_name.split('.')
    sequence = [None, 'half', 'quarter', 'one_eighth']
    for i in range(1, pyramid_level):
        output_image_array = pyramid[i]
        output_image = Image.fromarray(np.uint8(output_image_array))
        output_image.save(f'{file_name}_{sequence[i]}.{extension}')


if __name__ == "__main__":
    # # cross_correlation_2d
    # input_image = Image.open('Lena.png')
    # input_image_array = np.array(input_image)
    # box_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    # output_image_array = cross_correlation_2d(input_image_array, box_filter)
    # output_image = Image.fromarray(np.uint8(output_image_array))
    # output_image.save('Lena_Correlation.png')

    # # convolution_2d
    # input_image = Image.open('Lena.png')
    # input_image_array = np.array(input_image)
    # box_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    # output_image_array = cross_correlation_2d(input_image_array, box_filter)
    # output_image = Image.fromarray(np.uint8(output_image_array))
    # output_image.save('Lena_Convolution.png')

    # # gaussian_blur_kernel_2d
    # print(gaussian_blur_kernel_2d(3, 1))

    # # low_pass
    # input_image = Image.open('Lena.png')
    # input_image_array = np.array(input_image)
    # output_image_array = low_pass(input_image_array, 3, 1)
    # output_image = Image.fromarray(np.uint8(output_image_array))
    # output_image.save('Lena_Low_Pass.png')

    # # image_subsampling
    # input_image = Image.open('Lena.png')
    # input_image_array = np.array(input_image)
    # output_image_array = image_subsampling(input_image_array)
    # output_image = Image.fromarray(np.uint8(output_image_array))
    # output_image.save('Lena_Subsampling.png')

    # gaussian_pyramid
    output_gaussian_pyramid('Lena.png', 4)
    output_gaussian_pyramid('frog.jpg', 4)
    output_gaussian_pyramid('Yanami_Anna.jpg', 4)
