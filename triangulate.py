import time
import copy
from collections import defaultdict

import os
import shutil
import glob
import pickle

import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage import io

import cv2

from utils.storage import Landmark

def show_pcl(ax, pcl, title=None, colors=None):
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111, projection='3d')
    if title is not None:
        plt.title(title)
    if colors is not None:
        ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], 'o', color=colors/255., label='Pointcloud')
    else:
        ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], 'o', label='Pointcloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.tight_layout()
    plt.show();

def construct_intrinsic_matrix(focal_length, principal_point):
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    K[0,2] = principal_point[0]
    K[1,2] = principal_point[1]
    return K


def calculate_projection_matrix(local_R, local_t, prev_T, K):
    # local tansform
    T = np.eye(4)
    T[0:3, 0:3] = local_R
    T[0:3, 3:4] = local_t
    # accumulate transform
    T = prev_T @ T
    # make projection matrix
    R = T[0:3, 0:3]
    t = T[0:3, 3:4]
    P = np.zeros((3, 4))
    P[0:3, 0:3] = R.T  # rotation
    P[0:3, 3:4] = -R.T @ t  # true translation
    P = K @ P  # projection = intrinsic x extrinsic
    return T, P


def triangulate_3d_landmarks(sfm_storage, focal_length, principal_point, vis_map):
    """
    Triangulate 3d landmarks from 2d keypoints.
    
    vis_map:
        bool
        Whether to visuazlize 3d points on each iteration
        
    return:
        `SFMStorage` instace
        Filled with landmarks info
    """
    K = construct_intrinsic_matrix(focal_length, principal_point)
    print(f"Initial intrinsic camera matrix K = \n{K}")

    sfm_storage.img_pose[0].T = np.eye(4)  # transformation (extrinsic) matrix
    sfm_storage.img_pose[0].P = K @ np.eye(3, 4)  # projection matrix

    for i in range(len(sfm_storage.img_pose)-1):
        prev = sfm_storage.img_pose[i]
        cur = sfm_storage.img_pose[i+1]

        # keypoints on this image that are matched with
        # other points after robust matching algo (see prev section: "Feature matching")
        kp_used = [
            k for k in range(len(prev.kp)) 
            if prev.kp_match_exist(k, i+1)
        ]
        src = np.array([prev.kp[k] for k in kp_used])
        dst = np.array([cur.kp[prev.kp_matches[(k, i+1)]] for k in kp_used])
        print('src keypoints used:', src.shape)
        print('dest keypoints used:', dst.shape)

        # NOTE: pose from dst to src
        mask = np.empty(1)
        E, mask = cv2.findEssentialMat(
            dst, src, focal_length, principal_point, cv2.FM_RANSAC, 0.999, 1.0, mask  
        )  # !!! threshold may be lower
        print('after findEssentialMat:', mask.sum())
        local_R, local_t = np.empty(1), np.empty(1)
        retval, local_R, local_t, mask = cv2.recoverPose(
            E, dst, src, local_R, local_t, focal_length, principal_point, mask
        )
        print('after recoverPose:', mask.sum())
        cur.T, cur.P = calculate_projection_matrix(local_R, local_t, prev.T, K)

        # calculate homogeneous coordinates of 3D points
        points4D = np.zeros((4, len(src)))
        # !!! not dst->src?
        points4D = cv2.triangulatePoints(prev.P, cur.P, src.T, dst.T, points4D)

        # Find good triangulated points
        for j in range(len(kp_used)):
            if mask[j][0]:
                k = kp_used[j]
                match_idx = prev.kp_matches[(k, i+1)]
                # homogeneous coordinate to 3d point
                pt3d = points4D[:3, j] / points4D[3, j]
                if prev.kp_3d_exist(k):
                    # found a match with an existing landmark
                    landmark_id = prev.kp_landmark[k]
                    cur.kp_landmark[match_idx] = landmark_id
                    sfm_storage.landmark[landmark_id].pt += pt3d
                    pixel_coords_cur = np.array([cur.kp[match_idx][1], cur.kp[match_idx][0]], dtype=int)
                    sfm_storage.landmark[landmark_id].color += cur.img[pixel_coords_cur[0], pixel_coords_cur[1]]
                    sfm_storage.landmark[landmark_id].seen += 1
                else:
                    # add new 3d point
                    landmark = Landmark()
                    landmark.pt = pt3d
                    pixel_coords_prev = np.array([prev.kp[k][1], prev.kp[k][0]], dtype=int)
                    # landmark.color = prev.img[pixel_coords_prev[0], pixel_coords_prev[1]]
                    pixel_coords_cur = np.array([cur.kp[match_idx][1], cur.kp[match_idx][0]], dtype=int)
                    landmark.color = cur.img[pixel_coords_cur[0], pixel_coords_cur[1]]
                    landmark.seen = 2
                    sfm_storage.landmark.append(landmark)
                    new_landmark_id = len(sfm_storage.landmark) - 1
                    prev.kp_landmark[k] = new_landmark_id
                    cur.kp_landmark[match_idx] = new_landmark_id

        if vis_map:
            pcl = np.array([x.pt for x in sfm_storage.landmark])
            colors = np.array([x.color for x in sfm_storage.landmark])
            for e in range(len(sfm_storage.landmark)):
                if sfm_storage.landmark[e].seen >= 3:
                    pcl[e] = pcl[e] / (sfm_storage.landmark[e].seen - 1)
                # colors[e] = colors[e] / sfm_storage.landmark[e].seen
                colors[e] = [int(colors[e][0]), int(colors[e][1]), int(colors[e][2])]
            show_pcl(pcl, title=f'Num points: {pcl.shape[0]}', colors=colors)

    # Average out the landmarks's 3d position
    for l in sfm_storage.landmark:
        if l.seen >= 3:
            l.pt = l.pt / (l.seen - 1)
        l.color = l.color / l.seen
        l.color = [int(l.color[0]), int(l.color[1]), int(l.color[2])]
    
    return sfm_storage
