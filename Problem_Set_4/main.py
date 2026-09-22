import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import trimesh
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple
import time


def normalize_disparity_map(disparity_map):
    '''Normalize disparity map for visualization 
    disparity should be larger than zero
    '''
    return np.maximum(disparity_map, 0.0) / (disparity_map.max() + 1e-10)


def visualize_disparity_map(disparity_map, gt_map, save_path=None):
    '''Visualize or save disparity map and compare with ground truth
    '''
    # Normalize disparity maps
    disparity_map = normalize_disparity_map(disparity_map)
    gt_map = normalize_disparity_map(gt_map)
    # Visualize or save to file
    if save_path is None:
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imshow(concat_map, 'gray')
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imsave(save_path, concat_map, cmap='gray')


def task1_compute_disparity_map_simple(
    ref_img: np.ndarray,        # shape (H, W)
    sec_img: np.ndarray,        # shape (H, W)
    window_size: int, 
    disparity_range: Tuple[int, int],   # (min_disparity, max_disparity)
    matching_function: str      # can be 'SSD', 'SAD', 'normalized_correlation'
):
    '''Assume image planes are parallel to each other
    Compute disparity map using simple stereo system following the steps:
    1. For each row, scan all pixels in that row
    2. Generate a window for each pixel in ref_img
    3. Search for a disparity (d) within (min_disparity, max_disparity) in sec_img 
    4. Select the best disparity that minimize window difference between ref_img[row, col] and sec_img[row, col - d]
    '''
    # time_start = time.time()
    disparity_map = np.zeros_like(ref_img)
    H, W = ref_img.shape
    radius = window_size // 2
    min_disparity, max_disparity = disparity_range
    # 0. Pad ref_img and sec_img for window extraction
    ref_img = np.pad(ref_img, ((radius, radius), (radius, radius)), 'edge')
    sec_img = np.pad(sec_img, ((radius, radius), (radius, radius)), 'edge')
    # 1. For each row, scan all pixels in that row
    for row in range(radius, radius + H):
        for col in range(radius, radius + W):
            # 2. Generate a window for each pixel in ref_img
            ref_window = ref_img[row-radius:row+radius, col-radius:col+radius]
            best_disparity = min_disparity
            if matching_function == 'SSD' or matching_function == 'SAD':
                min_window_difference = float('inf')
            elif matching_function == 'normalized_correlation':
                max_normalized_correlation = -1
            # 3. Search for a disparity (d) within (min_disparity, max_disparity) in sec_img
            for d in range(min_disparity, max_disparity + 1):
                if col - d < radius:
                    break
                sec_window = sec_img[row-radius:row+radius, col-d-radius:col-d+radius]
                # 4. Select the best disparity that minimize window difference
                # between ref_img[row, col] and sec_img[row, col - d]
                if matching_function == 'SSD':    # sum of squared differences
                    cur_window_difference = np.sum((ref_window - sec_window) ** 2)
                    if cur_window_difference < min_window_difference:
                        min_window_difference = cur_window_difference
                        best_disparity = d
                elif matching_function == 'SAD':  # sum of absolute differences
                    cur_window_difference = np.sum(np.abs(ref_window - sec_window))
                    if cur_window_difference < min_window_difference:
                        min_window_difference = cur_window_difference
                        best_disparity = d
                elif matching_function == 'normalized_correlation':
                    cur_normalized_correlation = \
                        np.sum((ref_window - np.mean(ref_window)) * (sec_window - np.mean(sec_window))) / \
                        (np.sqrt(np.sum((ref_window - np.mean(ref_window)) ** 2)) *
                         np.sqrt(np.sum((sec_window - np.mean(sec_window)) ** 2)))
                    if cur_normalized_correlation > max_normalized_correlation:
                        max_normalized_correlation = cur_normalized_correlation
                        best_disparity = d
            disparity_map[row-radius, col-radius] = best_disparity
    # time_end = time.time()
    # with open("task_1_time_record.txt", 'a', encoding='utf-8') as f:
    #     f.write(f"Task 1 running time (window_size={window_size}, disparity_range={disparity_range}, "
    #             f"matching_function={matching_function}): {time_end - time_start}\n")
    return disparity_map


def task1_simple_disparity(ref_img, sec_img, gt_map, img_name='tsukuba'):
    '''Compute disparity maps for different settings
    '''
    window_sizes = [10]  # Try different window sizes
    disparity_range = (0, 20)  # Determine appropriate disparity range
    matching_functions = ['SSD']  # Try diftferent matching functions
    
    disparity_maps = []
    
    # Generate disparity maps for different settings
    for window_size in window_sizes:
        for matching_function in matching_functions:
            print(f"Computing disparity map for window_size={window_size}, disparity_range={disparity_range}, matching_function={matching_function}")
            disparity_map = task1_compute_disparity_map_simple(
                ref_img, sec_img, 
                window_size, disparity_range, matching_function)
            disparity_maps.append((disparity_map, window_size, matching_function, disparity_range))
            dmin, dmax = disparity_range
            visualize_disparity_map(
                disparity_map, gt_map, 
                save_path=f"output/task1_{img_name}_{window_size}_{dmin}_{dmax}_{matching_function}.png")
    return disparity_maps


def task2_compute_depth_map(disparity_map, baseline, focal_length):
    '''Compute depth map by z = fB / (x - x')
    Note that a disparity less or equal to zero should be ignored (set to zero) 
    '''
    mask = disparity_map > 0
    depth_map = np.zeros(disparity_map.shape)
    depth_map[mask] = focal_length * baseline / disparity_map[mask]
    return depth_map


def task2_visualize_pointcloud(
    ref_img: np.ndarray,        # shape (H, W, 3) 
    disparity_map: np.ndarray,  # shape (H, W)
    save_path: str = 'output/task2_tsukuba.ply'
):
    '''Visualize 3D pointcloud from disparity map following the steps:
    1. Calculate depth map from disparity
    2. Set pointcloud's XY as image's XY and and pointcloud's Z as depth
    3. Set pointcloud's color as ref_img's color
    4. Save pointcloud to ply files for visualizationh. We recommend to open ply file with MeshLab
    5. Adjust the baseline and focal_length for better performance
    6. You may need to cut some outliers for better performance
    '''
    baseline = 10
    focal_length = 10
    depth_map = task2_compute_depth_map(disparity_map, baseline, focal_length)

    # cut some outliers for better performance
    depth_map_tilda = depth_map.copy()
    mean = np.mean(depth_map_tilda)
    std = np.std(depth_map_tilda)
    depth_map_tilda = np.abs((depth_map_tilda - mean) / std)
    depth_map[depth_map_tilda > 2] = 0
    mask = depth_map > 0

    y_coords, x_coords = np.indices(depth_map.shape)
    # e.g.
    # grid = np.indices((2, 3))
    # grid[0] = [[0, 0, 0],
    #            [1, 1, 1]]  # row indices
    # grid[1] = [[0, 1, 2],
    #            [0, 1, 2]]  # column indices
    depth_map = depth_map / np.max(depth_map) * 200.0

    # Points
    points = np.hstack((x_coords[mask].reshape(-1, 1), y_coords[mask].reshape(-1, 1), depth_map[mask].reshape(-1, 1)))

    # Colors
    colors = ref_img[mask] / 255.0

    # Save pointcloud to ply file
    pointcloud = trimesh.PointCloud(points, colors)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pointcloud.export(save_path, file_type='ply')


def task3_compute_disparity_map_dp(ref_img, sec_img):
    ''' Conduct stereo matching with dynamic programming
    '''
    time_start = time.time()

    H, W = ref_img.shape
    window_size = 3
    radius = window_size // 2
    max_disparity = 16
    occlusionConstant = 256.0
    disparity_map_dp = np.zeros((H, W))
    ref_img = np.pad(ref_img, ((radius, radius), (radius, radius)), 'edge')
    sec_img = np.pad(sec_img, ((radius, radius), (radius, radius)), 'edge')
    for row in range(radius, radius + H):
        e = np.full((W, W), float('inf'))  # the matching cost of each match
        C = np.full((W, W), float('inf'))  # C[i, j]: the minimum cost to the match (i, j)
        B = np.zeros((W, W))  # leverage the matrix B to recover the path via backtracing
        # Compute matrix e according to SSD
        for i in range(radius, radius + W):
            sec_window = sec_img[row -radius:row+radius, i-radius:i+radius]
            for j in range(i, min(i + max_disparity, radius + W)):
                ref_window = ref_img[row-radius:row+radius, j-radius:j+radius]
                e[i-radius, j-radius] = np.sum((ref_window - sec_window) ** 2)
        # Initialize the first row
        C[0, 0] = min(e[0, 0], occlusionConstant)
        B[0, 0] = -1
        for j in range(1, W):
            C[0, j] = j * occlusionConstant
            B[0, j] = 3  # Occluded from the left
        # Fill the cost table with dynamic programming
        for i in range(1, W):
            for j in range(i, min(i + max_disparity, W)):
                min1 = C[i-1, j-1] + e[i, j]
                min2 = C[i-1, j] + occlusionConstant
                min3 = C[i, j-1] + occlusionConstant
                C[i, j] = min(min1, min2, min3)
                if min1 == C[i, j]:
                    B[i, j] = 1  # Matched
                elif min2 == C[i, j]:
                    B[i, j] = 2  # Occluded from the right
                elif min3 == C[i, j]:
                    B[i, j] = 3  # Occluded from the left
        # Recover the path via backtrac
        i, j = W - 1, W - 1
        while i >= 0 and j >= 0:
            if B[i, j] == 1:     # Matched
                disparity_map_dp[row-radius, j] = j - i
                i, j = i - 1, j - 1
            elif B[i, j] == 2:   # Occluded from the right
                disparity_map_dp[row - radius, j] = -2
                i = i - 1
            elif B[i, j] == 3:   # Occluded from the left
                disparity_map_dp[row - radius, j] = -3
                j = j - 1
            elif B[i, j] == -1:  #
                disparity_map_dp[row-radius, j] = 0
                break
        # Occlusion Filling
        for col in range(radius, radius + W):
            if disparity_map_dp[row-radius, col-radius] == -2:  # Occluded from the right
                for nearest_right_column in range(col, radius + W):
                    if disparity_map_dp[row-radius, nearest_right_column-radius] > 0:
                        disparity_map_dp[row-radius, col-radius] = disparity_map_dp[row-radius, nearest_right_column-radius]
                        break
            if disparity_map_dp[row-radius, col-radius] == -3:  # Occluded from the left
                for nearest_left_column in range(col, radius, -1):
                    if disparity_map_dp[row-radius, nearest_left_column-radius] > 0:
                        disparity_map_dp[row-radius, col-radius] = disparity_map_dp[row-radius, nearest_left_column-radius]
                        break

    time_end = time.time()
    print("task3 time cost:", time_end - time_start)
    return disparity_map_dp


def main(tasks): 
    
    # Read images and ground truth disparity maps
    moebius_img1 = cv2.imread("data/moebius1.png")
    moebius_img1_gray = cv2.cvtColor(moebius_img1, cv2.COLOR_BGR2GRAY)
    moebius_img2 = cv2.imread("data/moebius2.png")
    moebius_img2_gray = cv2.cvtColor(moebius_img2, cv2.COLOR_BGR2GRAY)
    moebius_gt = cv2.imread("data/moebius_gt.png", cv2.IMREAD_GRAYSCALE)

    tsukuba_img1 = cv2.imread("data/tsukuba1.jpg")
    tsukuba_img1_gray = cv2.cvtColor(tsukuba_img1, cv2.COLOR_BGR2GRAY)
    tsukuba_img2 = cv2.imread("data/tsukuba2.jpg")
    tsukuba_img2_gray = cv2.cvtColor(tsukuba_img2, cv2.COLOR_BGR2GRAY)
    tsukuba_gt = cv2.imread("data/tsukuba_gt.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Task 0: Visualize cv2 Results
    if '0' in tasks:   
        # Compute disparity maps using cv2
        stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)
        moebius_disparity_cv2 = stereo.compute(moebius_img1_gray, moebius_img2_gray)
        visualize_disparity_map(moebius_disparity_cv2, moebius_gt)
        tsukuba_disparity_cv2 = stereo.compute(tsukuba_img1_gray, tsukuba_img2_gray)
        visualize_disparity_map(tsukuba_disparity_cv2, tsukuba_gt)

        if '2' in tasks:
            print('Running task2 with cv2 results ...')
            task2_visualize_pointcloud(tsukuba_img1, tsukuba_disparity_cv2, save_path='output/task2_tsukuba_cv2.ply')

    ######################################################################
    # Note. Running on moebius may take a long time with your own code   #
    # In this homework, you are allowed only to deal with tsukuba images #
    ######################################################################

    # Task 1: Simple Disparity Algorithm
    if '1' in tasks:
        print('Running task1 ...')
        disparity_maps = task1_simple_disparity(tsukuba_img1_gray, tsukuba_img2_gray, tsukuba_gt, img_name='tsukuba')
        
        #####################################################
        # If you want to run on moebius images,             #
        # parallelizing with multiprocessing is recommended #
        #####################################################
        # task1_simple_disparity(moebius_img1_gray, moebius_img2_gray, moebius_gt, img_name='moebius')
        
        if '2' in tasks:
            print('Running task2 with disparity maps from task1 ...')
            for (disparity_map, window_size, matching_function, disparity_range) in disparity_maps:
                dmin, dmax = disparity_range
                task2_visualize_pointcloud(
                    tsukuba_img1, disparity_map, 
                    save_path=f'output/task2_tsukuba_{window_size}_{dmin}_{dmax}_{matching_function}.ply')      
        
    # Task 3: Non-local constraints
    if '3' in tasks:
        print('----------------- Task 3 -----------------')
        tsukuba_disparity_dp = task3_compute_disparity_map_dp(tsukuba_img1_gray, tsukuba_img2_gray)
        visualize_disparity_map(tsukuba_disparity_dp, tsukuba_gt, save_path='output/task3_tsukuba.png')
        
        if '2' in tasks:
            print('Running task2 with disparity maps from task3 ...')
            task2_visualize_pointcloud(tsukuba_img1, tsukuba_disparity_dp, save_path='output/task2_tsukuba_dp.ply')


if __name__ == '__main__':
    # Set tasks to run
    parser = argparse.ArgumentParser(description='Homework 4')
    parser.add_argument('--tasks', type=str, default='23')
    args = parser.parse_args()

    main(args.tasks)
