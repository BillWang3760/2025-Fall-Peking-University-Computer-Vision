from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy


def visualize_matches(I1, I2, matches):
    # display two images side-by-side with matches
    # this code is to help you visualize the matches, you don't need
    # to use it to produce the results for the assignment

    I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
    I3[:,:I1.size[0],:] = I1
    I3[:,I1.size[0]:,:] = I2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3).astype(np.uint8))
    ax.plot(matches[:,0],matches[:,1],  '+r')
    ax.plot( matches[:,2] + I1.size[0], matches[:,3], '+r')
    ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
    plt.show()


def normalize_points(pts):
    # Normalize points
    # 1. calculate mean and std
    # 2. build a transformation matrix
    # :return normalized_pts: normalized points
    # :return T: transformation matrix from original to normalized points

    # 1. calculate mean and std
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    # 2. build a transformation matrix
    T = np.array([[1 / std[0], 0, 0],
         [0, 1 / std[1], 0],
         [-mean[0] / std[0], -mean[1] / std[1], 1]])
    # 3. transform cartesian coordinates to homogeneous coordinates
    homo_pts = np.ones((pts.shape[0], 3))
    homo_pts[:, 0:2] = pts
    # 4. center the set of points at the origin
    # and scale it so [the mean squared distance between the origin and the point] is 2 pixels
    normalized_pts = np.dot(homo_pts, T)[:, 0:2]
    return normalized_pts, T


def fit_fundamental(matches):
    # Calculate fundamental matrix from grou nd truth matches
    # 1. (normalize points if necessary)
    # 2. (x2, y2, 1) * F * (x1, y1, 1)^T = 0 -> AX = 0
    # X = (f_11, f_12, ..., f_33) 
    # build A(N x 9) from matches(N x 4) according to Eight-Point Algorithm
    # 3. use SVD (np.linalg.svd) to decomposite the matrix
    # 4. take the smallest eigen vector(9, ) as F(3 x 3)
    # 5. use SVD to decomposite F, set the smallest eigenvalue as 0, and recalculate F
    # 6. Report your fundamental matrix results

    n = matches.shape[0]
    # 1. normalize the points and convert the coordinates to homogeneous coordinates
    pts_1 = matches[:, 0:2]  # matches[:, :2] is a point in the first image
    pts_2 = matches[:, 2:4]  # matches[:, 2:] is a corresponding point in the second image
    normalized_pts_1, T_1 = normalize_points(pts_1)
    normalized_pts_2, T_2 = normalize_points(pts_2)
    homo_pts_1 = np.ones((n, 3))
    homo_pts_1[:, 0:2] = normalized_pts_1
    homo_pts_2 = np.ones((n, 3))
    homo_pts_2[:, 0:2] = normalized_pts_2
    # # unnormalized algorithm
    # homo_pts_1 = np.ones((n, 3))
    # homo_pts_1[:, 0:2] = pts_1
    # homo_pts_2 = np.ones((n, 3))
    # homo_pts_2[:, 0:2] = pts_2
    # 2. form matrix A
    A = np.zeros((n, 9))
    A[:, 0:3] = homo_pts_2[:, 0:1] * homo_pts_1
    A[:, 3:6] = homo_pts_2[:, 1:2] * homo_pts_1
    A[:, 6:9] = homo_pts_1
    # 3. use SVD to find the eigenvector of ATA with the smallest eigenvalue
    U, S, V = np.linalg.svd(A)
    F_init = np.reshape(V[np.argmin(S)], (3, 3))
    # 4. enforce the rank-2 constraint
    U_F, S_F, V_F = np.linalg.svd(F_init)
    S_F_tilda = np.diag(S_F)
    S_F_tilda[np.argmin(S_F)] = 0
    F = np.dot(U_F, np.dot(S_F_tilda, V_F))
    # 5. transform fundamental matrix back to original units
    F = np.dot(T_2, np.dot(F, np.transpose(T_1)))
    return F


def visualize_fundamental(matches, F, I1, I2):
    # Visualize the fundamental matrix in image 2
    N = len(matches)
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1, np.kron(np.ones((3,1)), l).transpose())   # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis = 1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2],np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]] * 10    # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2).astype(np.uint8))
    ax.plot(matches[:, 2],matches[:, 3],  '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]],[matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]],[pt1[:, 1], pt2[:, 1]], 'g')
    plt.show()


def evaluate_fundamental(matches, F):
    N = len(matches)
    points1, points2 = matches[:, :2], matches[:, 2:]
    points1_homogeneous = np.concatenate([points1, np.ones((N, 1))], axis=1)
    points2_homogeneous = np.concatenate([points2, np.ones((N, 1))], axis=1)
    product = np.dot(np.dot(points2_homogeneous, F), points1_homogeneous.T)
    diag = np.diag(product)
    residual = np.mean(diag ** 2)
    return residual


## Task 0: Load data and visualize
## load images and match files for the first example
## matches[:, :2] is a point in the first image
## matches[:, 2:] is a corresponding point in the second image

library_image1 = Image.open('data/library1.jpg')
library_image2 = Image.open('data/library2.jpg')
library_matches = np.loadtxt('data/library_matches.txt')

lab_image1 = Image.open('data/lab1.jpg')
lab_image2 = Image.open('data/lab2.jpg')
lab_matches = np.loadtxt('data/lab_matches.txt')

## Visualize matches
visualize_matches(library_image1, library_image2, library_matches)
visualize_matches(lab_image1, lab_image2, lab_matches)

## Task 1: Fundamental matrix
## display second image with epipolar lines reprojected from the first image

# first, fit fundamental matrix to the matches
# Report your fundamental matrices, visualization and evaluation results
library_F = fit_fundamental(library_matches) # this is a function that you should write
visualize_fundamental(library_matches, library_F, library_image1, library_image2)
assert evaluate_fundamental(library_matches, library_F) < 0.5

lab_F = fit_fundamental(lab_matches) # this is a function that you should write
visualize_fundamental(lab_matches, lab_F, lab_image1, lab_image2)
assert evaluate_fundamental(lab_matches, lab_F) < 0.5

## Task 2: Camera Calibration


def calc_projection(points_2d, points_3d):
    # Calculate camera projection matrices
    # 1. Points_2d = P * Points_3d -> AX = 0
    # X = (p_11, p_12, ..., p_34) is flatten of P
    # build matrix A(2*N, 12) from points_2d
    # 2. SVD decomposite A
    # 3. take the eigen vector(12, ) of smallest eigen value
    # 4. return projection matrix(3, 4)
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return P: projection matrix

    n = points_2d.shape[0]
    # 1. convert the coordinates to homogeneous coordinates
    homo_pts_2d = np.ones((n, 3))
    homo_pts_2d[:, 0:2] = points_2d
    homo_pts_3d = np.ones((n, 4))
    homo_pts_3d[:, 0:3] = points_3d
    # 2. form matrix A
    A = np.zeros((2 * n, 12))
    A[0:2*n:2, 0:4] = homo_pts_3d
    A[1:2*n:2, 4:8] = homo_pts_3d
    A[0:2*n:2, 8:12] = -homo_pts_2d[:, 0:1] * homo_pts_3d
    A[1:2*n:2, 8:12] = -homo_pts_2d[:, 1:2] * homo_pts_3d
    # 3. use SVD to find the eigenvector of ATA with the smallest eigenvalue
    U, S, V = np.linalg.svd(A)
    P = np.reshape(V[np.argmin(S)], (3, 4))
    return P


def rq_decomposition(P):
    # Use RQ decomposition to calculte K, R, T
    # 1. perform RQ decomposition on left-most 3x3 matrix of P(3 x 4) to get K, R
    # 2. calculate T by P = K[R|T]
    # 3. normalize to set K[2, 2] = 1
    # :param P: projection matrix
    # :return K, R, T: camera matrices
    P_left = P[:, 0:3]
    P_right = P[:, 3:4]
    K, R = scipy.linalg.rq(P_left)         # perform RQ decomposition on left-most 3x3 matrix of P(3 x 4) to get K, R
    T = np.dot(np.linalg.inv(K), P_right)  # calculate T by P = K[R|T]
    K = K / K[2, 2]                        # normalize to set K[2, 2] = 1
    return K, R, T


def evaluate_points(P, points_2d, points_3d):
    # Visualize the actual 2D points and the projected 2D points calculated from
    # the projection matrix
    # You do not need to modify anything in this function, although you can if you
    # want to
    # :param P: projection matrix 3 x 4
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return points_3d_proj: project 3D points to 2D by P
    # :return residual: residual of points_3d_proj and points_2d

    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(P, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual


def triangulate_points(P1, P2, point1, point2):
    # Use linear least squares to triangulation 3d points
    # 1. Solve: point1 = P1 * point_3d
    #           point2 = P2 * point_3d
    # 2. use SVD decomposition to solve linear equations
    # :param P1, P2 (3 x 4): projection matrix of two cameras
    # :param point1, point2: points in two images
    # :return point_3d: 3D points calculated by triangulation
    A = np.zeros((4, 4))
    A[0] = P1[0] - point1[0] * P1[2]
    A[1] = P1[1] - point1[1] * P1[2]
    A[2] = P2[0] - point2[0] * P2[2]
    A[3] = P2[1] - point2[1] * P2[2]
    U, S, V = np.linalg.svd(A)
    homo_point_3d = V[np.argmin(S)]
    point_3d = homo_point_3d[0:3] / homo_point_3d[3]
    return point_3d


lab_points_3d = np.loadtxt('data/lab_3d.txt')

projection_matrix = dict()
for key, points_2d in zip(["lab_a", "lab_b"], [lab_matches[:, :2], lab_matches[:, 2:]]):
    P = calc_projection(points_2d, lab_points_3d)
    points_3d_proj, residual = evaluate_points(P, points_2d, lab_points_3d)
    distance = np.mean(np.linalg.norm(points_2d - points_3d_proj))
    # Check: residual should be < 20 and distance should be < 4
    assert residual < 20.0 and distance < 4.0
    projection_matrix[key] = P


## Task 3
## Camera Centers
projection_library_a = np.loadtxt('data/library1_camera.txt')
projection_library_b = np.loadtxt('data/library2_camera.txt')
projection_matrix["library_a"] = projection_library_a
projection_matrix["library_b"] = projection_library_b

for P in projection_matrix.values():
    # Paste your K, R, T results in your report
    K, R, T = rq_decomposition(P)


## Task 4: Triangulation
lab_points_3d_estimated = []
for point_2d_a, point_2d_b, point_3d_gt in zip(lab_matches[:, :2], lab_matches[:, 2:], lab_points_3d):
    point_3d_estimated = triangulate_points(projection_matrix['lab_a'], projection_matrix['lab_b'], point_2d_a, point_2d_b)

    # Residual between ground truth and estimated 3D points
    residual_3d = np.sum(np.linalg.norm(point_3d_gt - point_3d_estimated))
    assert residual_3d < 0.1
    lab_points_3d_estimated.append(point_3d_estimated)

# Residual between re-projected and observed 2D points
lab_points_3d_estimated = np.stack(lab_points_3d_estimated)
_, residual_a = evaluate_points(projection_matrix['lab_a'], lab_matches[:, :2], lab_points_3d_estimated)
_, residual_b = evaluate_points(projection_matrix['lab_b'], lab_matches[:, 2:], lab_points_3d_estimated)
assert residual_a < 20 and residual_b < 20

library_points_3d_estimated = []
for point_2d_a, point_2d_b in zip(library_matches[:, :2], library_matches[:, 2:]):
    point_3d_estimated = triangulate_points(projection_matrix['library_a'], projection_matrix['library_b'], point_2d_a, point_2d_b)
    library_points_3d_estimated.append(point_3d_estimated)

# Residual between re-projected and observed 2D points
library_points_3d_estimated = np.stack(library_points_3d_estimated)
_, residual_a = evaluate_points(projection_matrix['library_a'], library_matches[:, :2], library_points_3d_estimated)
_, residual_b = evaluate_points(projection_matrix['library_b'], library_matches[:, 2:], library_points_3d_estimated)
assert residual_a < 30 and residual_b < 30


## Task 5: Fundamental matrix estimation without ground-truth matches
import cv2


def align_pair(pts_1, pts_2):
    n = len(pts_1)
    homo_pts_1 = np.hstack((pts_1, np.ones((n, 1))))
    homo_pts_2 = np.hstack((pts_2, np.ones((n, 1))))
    iter_num = 1000
    threshold = 1e-4
    largest_inliers_num = 0
    average_residual = 0
    est_F = np.zeros((3, 3))
    matches_inliers = []
    for i in range(iter_num):
        # randomly choose 8 samples, and fit the fundamental matrix F to those samples
        random_sample_indices = np.random.choice(n, 8, replace=False)
        random_pts_1 = [pts_1[j] for j in random_sample_indices]
        random_pts_2 = [pts_2[j] for j in random_sample_indices]
        matches = np.hstack((random_pts_1, random_pts_2))
        F = fit_fundamental(matches)
        # calculate residual and count the inliers
        residual = np.diag(np.dot(np.dot(homo_pts_2, F), np.transpose(homo_pts_1))) ** 2
        inliers_num = np.sum([residual < threshold])
        # choose the homo_matrix that has the largest set of inliers as out final est_homo
        if inliers_num > largest_inliers_num:
            largest_inliers_num = inliers_num
            average_residual = np.mean(residual[residual < threshold])
            est_F = F
            # prepare for displaying the inliers
            inliers_1 = pts_1[residual < threshold]
            inliers_2 = pts_2[residual < threshold]
            matches_inliers = np.hstack((inliers_1, inliers_2))
    return est_F, matches_inliers


def fit_fundamental_without_gt(image1, image2):
    # Calculate fundamental matrix without groundtruth matches
    # 1. convert the images to gray
    # 2. compute SIFT keypoints and descriptors
    # 3. match descriptors with Brute Force Matcher
    # 4. select good matches
    # 5. extract matched keypoints
    # 6. compute fundamental matrix with RANSAC
    # :param image1, image2: two-view images
    # :return fundamental_matrix
    # :return matches: selected matched keypoints

    # 1. convert the images to gray
    img_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # 2. compute SIFT keypoints and descriptors
    sift = cv2.SIFT_create()  # initialize the detector
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_1, None)  # "None" means no mask is used
    keypoints_2, descriptors_2 = sift.detectAndCompute(img_2, None)
    # 3. match descriptors with Brute Force Matcher
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
    # 4. select good matches
    good_matches = []
    for match in matches:
        if match[0].distance < 0.8 * match[1].distance:
            good_matches.append(match[0])
    # 4. extract matched keypoints
    # queryIdx points to the feature point indices in the first image
    # trainIdx points to the feature point indices in the second image
    pts_1 = np.int16([keypoints_1[match.queryIdx].pt for match in good_matches]).reshape(-1, 2)  # pt:(x,y)
    pts_2 = np.int16([keypoints_2[match.trainIdx].pt for match in good_matches]).reshape(-1, 2)
    # 5. compute fundamental matrix with RANSAC
    fundamental_matrix, matches = align_pair(pts_1, pts_2)
    return fundamental_matrix, matches


house_image1 = Image.open('data/house1.jpg')
house_image2 = Image.open('data/house2.jpg')

house_F, house_matches = fit_fundamental_without_gt(np.array(house_image1), np.array(house_image2))
visualize_fundamental(house_matches, house_F, house_image1, house_image2)
