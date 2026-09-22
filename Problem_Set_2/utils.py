import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir
import matplotlib.pyplot as plt


def gaussian_blur_kernel_2d(kernel_size, standard_deviation):
    p = kernel_size // 2
    gaussian_kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussian_kernel[i][j] = np.exp(-((i - p) * (i - p) + (j - p) * (j - p)) / (2 * standard_deviation * standard_deviation))
    gaussian_kernel /= np.sum(gaussian_kernel)
    return gaussian_kernel


def convert_2_homogeneous_coordinates(pixels):
    n = len(pixels)
    pixels_array = np.array(pixels)
    homo_pixels = np.ones((n, 3))
    homo_pixels[:, 0:2] = pixels_array
    return homo_pixels


# feature detectors and descriptors
def my_feature_matching(img_1, img_2, mode):
    if mode == 'SIFT':
        sift = cv2.SIFT_create()  # initialize the detector
        interest_points_1, descriptors_1 = sift.detectAndCompute(img_1, None)  # "None" means no mask is used
        interest_points_2, descriptors_2 = sift.detectAndCompute(img_2, None)
        # SIFT descriptor is floating-point type, so we use L2-distance for matching
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif mode == 'AKAZE':
        akaze = cv2.AKAZE_create()
        interest_points_1, descriptors_1 = akaze.detectAndCompute(img_1, None)
        interest_points_2, descriptors_2 = akaze.detectAndCompute(img_2, None)
        # AKAZE and BRISK descriptors are binary type, so we use Hamming distance for matching
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif mode == 'BRISK':
        brisk = cv2.BRISK_create()
        interest_points_1, descriptors_1 = brisk.detectAndCompute(img_1, None)
        interest_points_2, descriptors_2 = brisk.detectAndCompute(img_2, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # use brute-feaforce matcher for feature matching
    matches = matcher.match(descriptors_1, descriptors_2)
    # extract pixel coordinates of matching feature points from the two images
    # queryIdx points to the feature point indices in the first image
    # trainIdx points to the feature point indices in the second image
    pixels_1 = np.int16([interest_points_1[match.queryIdx].pt for match in matches]).reshape(-1, 2)  # pt: (x,y)
    pixels_2 = np.int16([interest_points_2[match.trainIdx].pt for match in matches]).reshape(-1, 2)
    return pixels_1, pixels_2


def my_stitch_blend(img_1, img_2, est_homo, mode):
    h1, w1, d1 = np.shape(img_1)  # d=3 RGB
    h2, w2, d2 = np.shape(img_2)

    # Step 1: Compute the boundaries of the blended image
    # project four corner pixels of img_1 with estimated homo-matrix
    p1 = est_homo.dot(np.array([0, 0, 1]))
    p2 = est_homo.dot(np.array([0, h1, 1]))
    p3 = est_homo.dot(np.array([w1, 0, 1]))
    p4 = est_homo.dot(np.array([w1, h1, 1]))
    # convert homogeneous coordinates back to Cartesian coordinates
    p1 = np.int16(p1 / p1[2])
    p2 = np.int16(p2 / p2[2])
    p3 = np.int16(p3 / p3[2])
    p4 = np.int16(p4 / p4[2])
    # the boundaries of the blended image should contain all pixels of img_2 and converted img_1
    x_min = min(0, p1[0], p2[0], p3[0], p4[0])
    x_max = max(w2, p1[0], p2[0], p3[0], p4[0])
    y_min = min(0, p1[1], p2[1], p3[1], p4[1])
    y_max = max(h2, p1[1], p2[1], p3[1], p4[1])

    # Step 2: Create coordinate grid for the new image plane
    x_range = np.arange(x_min, x_max + 1, 1)
    y_range = np.arange(y_min, y_max + 1, 1)
    x, y = np.meshgrid(x_range, y_range)
    x = np.float32(x)
    y = np.float32(y)

    # Step 3: Calculate inverse transformation
    # Calculate the inverse of the homography matrix
    homo_inv = np.linalg.pinv(est_homo)
    # Apply inverse homography transformation
    # to map each point of the new image plane back to the coordinate system of img_1
    trans_x = homo_inv[0, 0] * x + homo_inv[0, 1] * y + homo_inv[0, 2]
    trans_y = homo_inv[1, 0] * x + homo_inv[1, 1] * y + homo_inv[1, 2]
    trans_z = homo_inv[2, 0] * x + homo_inv[2, 1] * y + homo_inv[2, 2]
    # convert homogeneous coordinates back to Cartesian coordinates
    trans_x = trans_x / trans_z
    trans_y = trans_y / trans_z
    trans_x = trans_x.astype(np.float32)
    trans_y = trans_y.astype(np.float32)  # numerical format correction

    # Step 4: Use the calculated mapping to remap both images to the new image plane
    # use inverse warping to transform img_1 onto the new image plane
    est_img_1 = cv2.remap(img_1, trans_x, trans_y, cv2.INTER_LINEAR)
    # use identity mapping to transform img_2 onto the new image plane
    est_img_2 = cv2.remap(img_2, x, y, cv2.INTER_LINEAR)
    # initialize est_img
    est_img = np.zeros(np.shape(est_img_1))

    # alpha blending:
    if mode == 'alpha':
        # Create two full-1 matrices of the same size as the original image to represent the weight of each pixel
        alpha1 = cv2.remap(np.ones(np.shape(img_1)), trans_x, trans_y, cv2.INTER_LINEAR)
        alpha2 = cv2.remap(np.ones(np.shape(img_2)), x, y, cv2.INTER_LINEAR)
        # Overlapping areas have two weights added together, while non-overlapping areas have only one weight
        alpha = alpha1 + alpha2
        alpha[alpha == 0] = 2  # Why?
        # normalize the weights so that the sum of the two weights at each pixel position is 1
        alpha1 = alpha1 / alpha
        alpha2 = alpha2 / alpha
        # use normalized weights to blend two transformed images
        est_img = est_img_1 * alpha1 + est_img_2 * alpha2

    else:
        # Step 5: Determine the left image and the right image
        non_zero_1 = np.argwhere(est_img_1 > 0)
        non_zero_2 = np.argwhere(est_img_2 > 0)
        left_1, right_1 = np.min(non_zero_1[:, 1]), np.max(non_zero_1[:, 1])
        left_2, right_2 = np.min(non_zero_2[:, 1]), np.max(non_zero_2[:, 1])
        # if est_img_1 is not the left image, exchange est_img_1 and est_img_2 together with their boundaries
        if left_1 > left_2:
            est_img_1, est_img_2 = est_img_2, est_img_1
            left_1, left_2, right_1, right_2 = left_2, left_1, right_2, right_1

        # pyramid blending
        if mode == 'pyramid':
            height = 4
            # 1. Build Gaussian Pyramid and Laplacian Pyramid for est_img_1
            gaussian_pyramid_1 = [est_img_1.copy()]
            for i in range(height - 1):
                gaussian_pyramid_1.append(cv2.pyrDown(gaussian_pyramid_1[-1]))
            laplacian_pyramid_1 = []
            for i in range(height - 1):
                size = (gaussian_pyramid_1[i].shape[1], gaussian_pyramid_1[i].shape[0])
                upsampling = cv2.pyrUp(gaussian_pyramid_1[i+1], dstsize=size)
                laplacian = cv2.subtract(gaussian_pyramid_1[i], upsampling)
                laplacian_pyramid_1.append(laplacian)
            laplacian_pyramid_1.append(gaussian_pyramid_1[-1])
            # 2. Build Gaussian Pyramid and Laplacian Pyramid for est_img_2
            gaussian_pyramid_2 = [est_img_2.copy()]
            for i in range(height - 1):
                gaussian_pyramid_2.append(cv2.pyrDown(gaussian_pyramid_2[-1]))
            laplacian_pyramid_2 = []
            for i in range(height - 1):
                size = (gaussian_pyramid_2[i].shape[1], gaussian_pyramid_2[i].shape[0])
                upsampling = cv2.pyrUp(gaussian_pyramid_2[i+1], dstsize=size)
                laplacian = cv2.subtract(gaussian_pyramid_2[i], upsampling)
                laplacian_pyramid_2.append(laplacian)
            laplacian_pyramid_2.append(gaussian_pyramid_2[-1])
            # 3. Blend the Laplacian Pyramids
            est_img_pyramid = []
            for img_1_lap, img_2_lap in zip(laplacian_pyramid_1, laplacian_pyramid_2):
                cols = img_1_lap.shape[1]
                # take the first half of the left image, and take the latter half of the right image
                # join these two halves horizontally
                laplacian = np.hstack((img_1_lap[:, :cols//2], img_2_lap[:, cols//2:]))
                est_img_pyramid.append(laplacian)
            # 4. Reconstruct image from Laplacian Pyramid
            est_img = est_img_pyramid[-1]
            for i in range(height - 2, -1, -1):
                size = (est_img_pyramid[i].shape[1], est_img_pyramid[i].shape[0])
                upsampling = cv2.pyrUp(est_img, dstsize=size)
                est_img = cv2.add(est_img_pyramid[i], upsampling)

        # Poisson blending
        elif mode == 'poisson':
            # 1. Create a basic stitched image
            est_img[:, :right_1+1, :] = est_img_1[:, :right_1+1, :]  # put est_img_1 in the left region
            est_img[:, right_1:, :] = est_img_2[:, right_1:, :]      # put est_img_2 in the right region
            # 2. Determine the vertical boundary of the overlapping area
            y_1 = [p[0] for p in np.argwhere(est_img_1[:, left_2+1:right_1, :] > 0)]
            y_2 = [p[0] for p in np.argwhere(est_img_2[:, left_2+1:right_1, :] > 0)]
            up = max(np.max(y_1), np.max(y_2))
            down = min(np.min(y_1), np.min(y_2))
            # 3. Prepare parameters for seamlessClone
            src = np.uint8(est_img_1[down+1:up, left_2+1:right_1, :])  # source image: extract overlapping areas from the left image
            dst = np.uint8(est_img)  # target image
            mask = np.ones(np.shape(src), dtype=np.uint8) * 255      # create a full white mask: all pixels in the source image should be used
            pos = (int((left_2 + 1 + right_1) / 2), int((up + 1 + down) / 2) )  # the center position of the overlapping area
            # 4. Carry out seamlessClone with MIXED_CLONE
            est_img = cv2.seamlessClone(src, dst, mask, pos, cv2.MIXED_CLONE)

    return est_img
