# This is a raw framework for image stitching using Harris corner detection.
# For libraries you can use modules in numpy, scipy, cv2, os, etc.
import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir
import matplotlib.pyplot as plt
from utils import *


IMGDIR = 'Problem2Images'


def gradient_x(img):
    # convert img to grayscale
    # should we use int type to calclate gradient?
    # should we conduct some pre-processing to remove noise? which kernel should we pply?
    # which kernel should intwe choose to calculate gradient_x?
    # TODO
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray scale conversion
    gray_img = np.float64(gray_img)                   # uint8 to float64
    blur_img = ndimage.gaussian_filter(gray_img, sigma=3.0, mode='reflect')  # Gaussian blur
    grad_x = ndimage.sobel(blur_img, axis=1, mode='reflect')                 # Sobel operation
    return grad_x


def gradient_y(img):
    # TODO
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray scale conversion
    gray_img = np.float64(gray_img)                   # uint8 to float64
    blur_img = ndimage.gaussian_filter(gray_img, sigma=3.0, mode='reflect')  # Gaussian blur
    grad_y = ndimage.sobel(blur_img, axis=0, mode='reflect')                 # Sobel operation
    return grad_y


def harris_response(img, alpha, win_size):
    # In this function you are going to claculate harris response R.
    # Please refer to 04_Feature_Detection.pdf page 32 for details. 
    # You have to discover how to calculate det(M) and trace(M), and
    # remember to smooth the gradients. 
    # Avoid using too much "for" loops to speed up.
    # TODO
    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
    # In practice, the derivatives in M are convolved with a window w,
    # usually a Gaussian window to reduce noise and enhance the detection of corners.
    gaussian_kernel = gaussian_blur_kernel_2d(win_size, standard_deviation=5.0)
    A = ndimage.convolve(grad_x * grad_x, gaussian_kernel, mode='reflect')
    B = ndimage.convolve(grad_x * grad_y, gaussian_kernel, mode='reflect')
    C = ndimage.convolve(grad_y * grad_y, gaussian_kernel, mode='reflect')
    R = A * C - B ** 2 - alpha * (A + C) ** 2
    return R


def corner_selection(R, thresh, min_dist):
    # non-maximal suppression for R to get R_selection and transform selected corners to list of tuples
    # hint: 
    #   use ndimage.maximum_filter()  to achieve non-maximum suppression
    #   set those which aren’t **local maximum** to zero.
    # TODO
    R_thresh = R.copy()
    R_thresh[R_thresh < thresh] = 0
    R_max = ndimage.maximum_filter(R, size=min_dist)
    is_maxima = (R_max == R_thresh) & (R_thresh != 0)
    coords = np.argwhere(is_maxima)
    pix = [(coord[1], coord[0]) for coord in coords]
    return pix


def histogram_of_gradients(img, pix):
    # no template for coding, please implement by yourself.
    # You can refer to implementations on Github or other websites
    # Hint: 
    #   1. grad_x & grad_y
    #   2. grad_dir by arctan function
    #   3. for each interest point, choose m*m blocks with each consists of m*m pixels
    #   4. I divide the region into n directions (maybe 8).
    #   5. For each blocks, calculate the number of derivatives in those directions and normalize the Histogram. 
    #   6. After that, select the prominent gradient and take it as principle orientation.
    #   7. Then rotate it’s neighbor to fit principle orientation and calculate the histogram again. 
    # TODO
    features = []
    bins = np.arange(0, 360 + 1, 45)
    # 1. Calculate gradients
    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
    grad_dir = np.arctan2(grad_y, grad_x) * 180 / np.pi + 180  # np.arctan2: [-pi, pi], converted to [0, 360]
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # 2. Extract the region (4 blocks * 4 blocks, or 16 pixels * 16 pixels) around the interest point
    for (x, y) in pix:
        interest_point_feature = []
        region_x_min, region_x_max = max(0, x - 8), min(img.shape[1], x + 8)
        region_y_min, region_y_max = max(0, y - 8), min(img.shape[0], y + 8)
        region_dir = grad_dir[region_y_min:region_y_max, region_x_min:region_x_max]
        region_mag = grad_mag[region_y_min:region_y_max, region_x_min:region_x_max]
        # 3. Select the prominent gradient of the region and take it as principle orientation
        region_hist, _ = np.histogram(region_dir, bins=bins, weights=region_mag)
        principal_orientation = np.argmax(region_hist) * 45
        # 4. Rotate its neighbor to fit principle orientation
        region_dir = (region_dir - principal_orientation) % 360
        # 5. Extract the blocks (4 pixels * 4 pixels)
        for i in range(4):
            for j in range(4):
                block_x_min, block_x_max = 4 * i, min(region_dir.shape[1], 4 * (i + 1))
                block_y_min, block_y_max = 4 * j, min(region_dir.shape[0], 4 * (j + 1))
                if block_x_max <= block_x_min or block_y_max <= block_y_min:
                    interest_point_feature.extend([0 for _ in range(8)])
                    continue
                block_dir = region_dir[block_y_min:block_y_max, block_x_min:block_x_max]
                block_mag = region_mag[block_y_min:block_y_max, block_x_min:block_x_max]
                # 6. For each block, calculate the number of derivatives in those directions and normalize the Histogram
                block_hist, _ = np.histogram(block_dir, bins=bins, weights=block_mag)
                block_hist /= np.sqrt(np.sum(block_hist ** 2) + 1e-6)
                interest_point_feature.extend(block_hist)
        features.extend(interest_point_feature)
    features = np.array(features).reshape((len(pix), 128))
    return features


def feature_matching(img_1, img_2):
    R1 = harris_response(img_1, 0.04, 9)
    R2 = harris_response(img_2, 0.04, 9)
    cor1 = corner_selection(R1, 0.01*np.max(R1), 5)
    cor2 = corner_selection(R2, 0.01*np.max(R1), 5)
    fea1 = histogram_of_gradients(img_1, cor1)
    fea2 = histogram_of_gradients(img_2, cor2)
    dis = spatial.distance.cdist(fea1, fea2, metric='euclidean')
    threshold = 0.8
    pixels_1 = []
    pixels_2 = []
    p1, p2 = np.shape(dis)
    if p1 < p2:
        for p in range(p1):
            dis_min = np.min(dis[p])
            pos = np.argmin(dis[p])
            dis[p][pos] = np.max(dis)
            if dis_min/np.min(dis[p]) <= threshold:
                pixels_1.append(cor1[p])
                pixels_2.append(cor2[pos])
                dis[:, pos] = np.max(dis)

    else:
        for p in range(p2):
            dis_min = np.min(dis[:, p])
            pos = np.argmin(dis[:, p])
            dis[pos][p] = np.max(dis)
            if dis_min/np.min(dis[:, p]) <= threshold:
                pixels_2.append(cor2[p])
                pixels_1.append(cor1[pos])
                dis[pos] = np.max(dis)
    min_len = min(np.shape(cor1)[0], np.shape(cor2)[0])
    rate = np.shape(pixels_1)[0]/min_len
    assert rate >= 0.03, "Fail to Match!"
    return pixels_1, pixels_2


def test_matching():    
    img_1 = cv2.imread(f'{IMGDIR}/1_1.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/1_2.jpg')

    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    pixels_1, pixels_2 = feature_matching(img_1, img_2)

    H_1, W_1 = img_gray_1.shape
    H_2, W_2 = img_gray_2.shape

    img = np.zeros((max(H_1, H_2), W_1 + W_2, 3))
    img[:H_1, :W_1, (2, 1, 0)] = img_1 / 255
    img[:H_2, W_1:, (2, 1, 0)] = img_2 / 255
    
    plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(img)

    N = len(pixels_1)
    for i in range(N):
        x1, y1 = pixels_1[i]
        x2, y2 = pixels_2[i]
        plt.plot([x1, x2+W_1], [y1, y2])

    # plt.show()
    plt.savefig('local_feature_matching_1.jpg')


def compute_homography(pixels_1, pixels_2):
    # compute the best-fit homography using the Singular Value Decomposition (SVD)
    # homography matrix is a (3,3) matrix consisting rotation, translation and projection information.
    # consider how to form matrix A for U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    # homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    # TODO
    n = len(pixels_1)
    # convert the coordinates to homogeneous coordinates
    homo_pixels_1 = convert_2_homogeneous_coordinates(pixels_1)
    homo_pixels_2 = convert_2_homogeneous_coordinates(pixels_2)
    # form matrix A
    A = np.zeros((2 * n, 9))
    A[0:2*n:2, 0:3] = homo_pixels_1
    A[1:2*n:2, 3:6] = homo_pixels_1
    A[0:2*n:2, 6:9] = -homo_pixels_1 * homo_pixels_2[:, 0:1]
    A[1:2*n:2, 6:9] = -homo_pixels_1 * homo_pixels_2[:, 1:2]
    # use SVD to find the eigenvector of ATA with the smallest eigenvalue
    U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    return homo_matrix


def align_pair(pixels_1, pixels_2):
    # utilize \verb|homo_coordinates| for homogeneous pixels
    # and \verb|compute_homography| to calulate homo_matrix
    # implement RANSAC to compute the optimal alignment.
    # you can refer to implementations online.
    n = len(pixels_1)
    homo_pixels_1 = convert_2_homogeneous_coordinates(pixels_1)
    homo_pixels_2 = convert_2_homogeneous_coordinates(pixels_2)
    iter_num = 600
    threshold = 3.0
    largest_inliers_num = 0
    est_homo = np.zeros((3, 3))
    for i in range(iter_num):
        # randomly choose 4 samples, and fit the homography matrix H to those samples
        random_sample_indices = np.random.choice(n, 4, replace=False)
        random_pixels_1 = [pixels_1[j] for j in random_sample_indices]
        random_pixels_2 = [pixels_2[j] for j in random_sample_indices]
        homo_matrix = compute_homography(random_pixels_1, random_pixels_2)
        # calculate geometric error and count the inliers
        pixels_2_tilda = homo_pixels_1.dot(np.transpose(homo_matrix))
        homo_pixels_2_tilda = pixels_2_tilda / pixels_2_tilda[:, 2:3]
        geometric_error = spatial.distance.cdist(homo_pixels_2_tilda, homo_pixels_2, metric='euclidean')
        geometric_error = np.diagonal(geometric_error)
        inliers_num = np.sum([geometric_error < threshold])
        # choose the homo_matrix that has the largest set of inliers as out final est_homo
        if inliers_num > largest_inliers_num:
            largest_inliers_num = inliers_num
            est_homo = homo_matrix
    return est_homo


def stitch_blend(img_1, img_2, est_homo):
    # hint: 
    # First, project four corner pixels with estimated homo-matrix
    # and converting them back to Cartesian coordinates after normalization.
    # Together with four corner pixels of the other image, we can get the size of new image plane.
    # Then, remap both image to new image plane and blend two images using Alpha Blending.
    h1, w1, d1 = np.shape(img_1)  # d=3 RGB
    h2, w2, d2 = np.shape(img_2)

    # Step 1: Compute the boundaries of the blended image
    # project four corner pixels of img_1 with estimated homo-matrix
    p1 = est_homo.dot(np.array([0, 0, 1]))
    p2 = est_homo.dot(np.array([0, h1, 1]))
    p3 = est_homo.dot(np.array([w1, 0, 1]))
    p4 = est_homo.dot(np.array([w1, h1, 1]))
    # convert homogeneous coordinates back to Cartesian coordinates
    p1 = np.int16(p1/p1[2])
    p2 = np.int16(p2/p2[2])
    p3 = np.int16(p3/p3[2])
    p4 = np.int16(p4/p4[2])
    # the boundaries of the blended image should contain all pixels of img_2 and converted img_1
    x_min = min(0, p1[0], p2[0], p3[0], p4[0])
    x_max = max(w2, p1[0], p2[0], p3[0], p4[0])
    y_min = min(0, p1[1], p2[1], p3[1], p4[1])
    y_max = max(h2, p1[1], p2[1], p3[1], p4[1])

    # Step 2: Create coordinate grid for the new image plane
    x_range = np.arange(x_min, x_max+1, 1)
    y_range = np.arange(y_min, y_max+1, 1)
    x, y = np.meshgrid(x_range, y_range)
    x = np.float32(x)
    y = np.float32(y)

    # Step 3: Calculate inverse transformation
    # Calculate the inverse of the homography matrix
    homo_inv = np.linalg.pinv(est_homo)
    # Apply inverse homography transformation
    # to map each point of the new image plane back to the coordinate system of img_1
    trans_x = homo_inv[0, 0]*x+homo_inv[0, 1]*y+homo_inv[0, 2]
    trans_y = homo_inv[1, 0]*x+homo_inv[1, 1]*y+homo_inv[1, 2]
    trans_z = homo_inv[2, 0]*x+homo_inv[2, 1]*y+homo_inv[2, 2]
    # convert homogeneous coordinates back to Cartesian coordinates
    trans_x = trans_x/trans_z
    trans_y = trans_y/trans_z
    trans_x = trans_x.astype(np.float32)
    trans_y = trans_y.astype(np.float32)  # numerical format correction

    # Step 4: Use the calculated mapping to remap both images to the new image plane
    # use inverse warping to transform img_1 onto the new image plane
    est_img_1 = cv2.remap(img_1, trans_x, trans_y, cv2.INTER_LINEAR)
    # use identity mapping to transform img_2 onto the new image plane
    est_img_2 = cv2.remap(img_2, x, y, cv2.INTER_LINEAR)

    # Step 5: Alpha-blending
    # Create two full-1 matrices of the same size as the original image to represent the weight of each pixel
    alpha1 = cv2.remap(np.ones(np.shape(img_1)), trans_x,
                       trans_y, cv2.INTER_LINEAR)
    alpha2 = cv2.remap(np.ones(np.shape(img_2)), x, y, cv2.INTER_LINEAR)
    # Overlapping areas have two weights added together, while non-overlapping areas have only one weight
    alpha = alpha1+alpha2
    alpha[alpha == 0] = 2  # Why?
    # normalize the weights so that the sum of the two weights at each pixel position is 1
    alpha1 = alpha1/alpha
    alpha2 = alpha2/alpha
    # use normalized weights to blend two transformed images
    est_img = est_img_1*alpha1 + est_img_2*alpha2
    return est_img


def generate_panorama(ordered_img_seq):
    len = np.shape(ordered_img_seq)[0]
    mid = int(len/2) # middle anchor
    i = mid-1
    j = mid+1
    principle_img = ordered_img_seq[mid]
    while(j < len):
        pixels1, pixels2 = feature_matching(ordered_img_seq[j], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[j], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        j = j+1  
    while(i >= 0):
        pixels1, pixels2 = feature_matching(ordered_img_seq[i], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[i], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        i = i-1  
    est_pano = principle_img
    return est_pano


def my_generate_panorama(ordered_img_seq):
    length = np.shape(ordered_img_seq)[0]
    mid = int(length / 2)
    i = mid - 1
    j = mid + 1
    principle_img = ordered_img_seq[mid]
    while j < length:
        pixels_1, pixels_2 = my_feature_matching(ordered_img_seq[j], principle_img, mode="SIFT")
        homo_matrix = align_pair(pixels_1, pixels_2)
        principle_img = my_stitch_blend(ordered_img_seq[j], principle_img, homo_matrix, mode="pyramid")
        principle_img = np.uint8(principle_img)
        j = j + 1
    while i >= 0:
        pixels_1, pixels_2 = my_feature_matching(ordered_img_seq[i], principle_img, mode="SIFT")
        homo_matrix = align_pair(pixels_1, pixels_2)
        principle_img = my_stitch_blend(ordered_img_seq[i], principle_img, homo_matrix, mode="pyramid")
        principle_img = np.uint8(principle_img)
        i = i - 1
    est_pano = principle_img
    return est_pano


if __name__ == '__main__':
    # make image list
    # call generate panorama and it should work well
    # save the generated image following the requirements

    # test_matching()

    # # stitch_blend_test
    # img_1 = cv2.imread(f'{IMGDIR}/1_1.jpg')
    # img_2 = cv2.imread(f'{IMGDIR}/1_2.jpg')
    # pixels_1, pixels_2 = feature_matching(img_1, img_2)
    # est_homo = align_pair(pixels_1, pixels_2)
    # est_img = stitch_blend(img_1, img_2, est_homo)
    # est_img = np.uint8(est_img)
    # cv2.imwrite("blend_1.png", est_img)

    # an example
    # 1.grail
    img_1 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail00.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail01.jpg')
    img_3 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail02.jpg')
    img_4 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail03.jpg')
    img_5 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail04.jpg')
    # 2. library
    # img_1 = cv2.imread(f'{IMGDIR}/panoramas/library/9.jpg')
    # img_2 = cv2.imread(f'{IMGDIR}/panoramas/library/10.jpg')
    # img_3 = cv2.imread(f'{IMGDIR}/panoramas/library/11.jpg')
    # img_4 = cv2.imread(f'{IMGDIR}/panoramas/library/12.jpg')
    # img_5 = cv2.imread(f'{IMGDIR}/panoramas/library/13.jpg')
    # 3.parrington
    # img_1 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn00.jpg')
    # img_2 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn01.jpg')
    # img_3 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn02.jpg')
    # img_4 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn03.jpg')
    # img_5 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn04.jpg')
    # 4. Xue-Mountain-Entrance
    # img_1 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0171.jpg')
    # img_2 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0172.jpg')
    # img_3 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0173.jpg')
    # img_4 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0174.jpg')
    # img_5 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0175.jpg')
    img_list = []
    # img_list.append(img_1)
    # img_list.append(img_2)
    img_list.append(img_3)
    img_list.append(img_4)
    img_list.append(img_5)
    pano = my_generate_panorama(img_list)
    cv2.imwrite("panorama_1_pyramid_1.png", pano)
