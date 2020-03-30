import numpy as np

from skimage import io

from skimage.feature import ORB
from skimage.color import rgb2gray

from skimage.feature import match_descriptors
from skimage.transform import ProjectiveTransform, AffineTransform
from skimage.measure import ransac

from skimage.feature import plot_matches

import cv2

from storage import SFMStorage, ImagePose
import time
import matplotlib.pyplot as plt
from skimage.feature import plot_matches

def detect_kp_desc(img, method='orb', n_keypoints=2000, **args):
    """Find keypoints and their descriptors on the image.
    
    img:
        `np.array` of shape == WxHx3
        RGB image
    method:
        str
        Name of the method to use. Options are: ['orb', 'lf-net']
    n_keypoints:
        int
        Number of keypoints to find
    **args:
        dict
        Other parameters to pass to keypoints detector without any chanages
    
    return:
        tuple (2,)
        Coordinates and descriptors of found keypoints
    """

    if method == 'orb':
        detector_exctractor = ORB(n_keypoints=n_keypoints, **args)
#         detector_exctractor = cv2.ORB_create(nfeatures=n_keypoints, **args)
    elif method == 'lf-net':
#         https://github.com/vcg-uvic/lf-net-release
        raise NotImplemetedError()
    detector_exctractor.detect_and_extract(rgb2gray(img).astype(np.float64))
    return detector_exctractor.keypoints, detector_exctractor.descriptors


def match_flann(
    src_descriptors, dest_descriptors,
    num_trees=5,
    checks=50,
    n_neighbors=2,
    apply_ratio_test=False
):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=num_trees)
    search_params = dict(checks=checks)
    # matching procedure
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    src_descriptors = src_descriptors.astype(np.float32)
    dest_descriptors = dest_descriptors.astype(np.float32)
    matches = flann.knnMatch(src_descriptors, dest_descriptors, k=n_neighbors)
    # ratio test as per Lowe's paper
    if apply_ratio_test:
        good = []
        matchesMask = [[0,0] for i in xrange(len(matches))]
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i]=[1,0]
                good.append([m])
        matches = good
    # get rid of cv2 format
    matches = np.array([[
        matches[i][j].queryIdx, matches[i][j].trainIdx] 
        for i in range(len(matches)) 
        for j in range(len(matches[i]))
    ]) 
    return matches

            
def match_robust(
    src_keypoints, src_descriptors, 
    dest_keypoints, dest_descriptors,
    method='brute-force',
    model_class=AffineTransform, 
    min_samples=4, 
    residual_threshold=1, 
    max_trials=5000, 
    **kwargs
):
    """Find matches of keypoints between two images and filter outlier with RANSAC.
    
    src_keypoints, dest_keypoints:
        (both) `np.array(float)` of the shape (N, 2)
        Coordinates of keypoints in format (x, y) each
    src_descriptors, dest_descriptors:
        (both) `np.array(float)` of the shape (N, P) 
        Descriptors of the keypoints. P is a descriptor dim
     model_class:
         `skimage.transform`
         Model of plane transformation
     min_samples:
         int
         Number of points needed to construct the transformation.
         The harder the model is the more points are needed.
         See the docs to know about each particular model.
     residual_threshold:
         int
         Distance in pixel in which the points are considered as "the same".
     max_trials:
         int
         Number of trials to do before stopping.
         The more is the number of keypoints, the larger `max_trials` should be.
     **kwargs:
         dict
         Other params, kept unchanged
     
    
    return:
        model, matches
        model: transformation of the 1st frame to the second
        matches: 
            `np.array` of shape == (num_matched_keypoints, 2)
            List of matches in format: np.array([[src_kp_id1, dest_kp_id1], ...], dtype=int)
            
    """
    if method == 'brute-force':
        matches = match_descriptors(src_descriptors, dest_descriptors)
    elif method == 'flann':
        matches = match_descriptors(src_descriptors, dest_descriptors)
    model, inliers = ransac(
        (src_keypoints[matches[:,0]], dest_keypoints[matches[:,1]]), 
        model_class=model_class, 
        min_samples=min_samples, 
        residual_threshold=residual_threshold, 
        max_trials=max_trials, 
        **kwargs
    )
    matches = matches[inliers]
    return model, matches
    

def extract_features(image_collection_raw, image_collection, profile=True):
    """
    Match keypoints of each image pair by their descriptors.
    """
    sfm_storage = SFMStorage()
    for i in range(len(image_collection)):
        a = ImagePose()
        a.img = image_collection_raw[i]
        if profile: beg = time.time()
        img = image_collection[i]
        keypoints, a.desc = detect_kp_desc(img)
        a.kp = keypoints[:,::-1]  # !!! because we need (x,y), not (y,x)
        if profile: print('Detect time:', time.time() - beg)
        sfm_storage.img_pose.append(a)
    return sfm_storage


def match_pairwise(sfm_storage, vis_matches, profile=True):
    """
    Match keypoints of each image pair by their descriptors.
    
    sfm_storage:
        `SFMStorage` instance
    
    vis_matches:
        bool
        Whether to draw the matches
     
    profile:
        bool
        Whether to measure execution time
     
    return:
        `SFMStorage` instance
        `SFMStorage` instance filled with matches information
    """
    for i in range(len(sfm_storage.img_pose)):
        for j in range(i+1, len(sfm_storage.img_pose)):
            # detect features and extract descriptors
            src_keypoints, src_descriptors = sfm_storage.img_pose[i].kp, sfm_storage.img_pose[i].desc
            dest_keypoints, dest_descriptors = sfm_storage.img_pose[j].kp, sfm_storage.img_pose[j].desc
            # RANSAC outlier filtering
            if profile: beg = time.time()
            robust_transform, matches = match_robust(
                src_keypoints, src_descriptors,
                dest_keypoints, dest_descriptors, 
                method='flann', 
                min_samples=4, 
                residual_threshold=100, 
                max_trials=3000, 
            )
            if profile: print('Match and RANSAC time:', time.time() - beg)
            # save img1-kp1-img2-kp2 matches to global helper SFM instance
            for m in matches:
                sfm_storage.img_pose[i].kp_matches[(m[0], j)] = m[1]
                sfm_storage.img_pose[j].kp_matches[(m[1], i)] = m[0]
            print(f"Feature matching: image {i} <-> image {j} ==> {len(matches)} matches")
            # vis
            if vis_matches:
                plt.figure()
                ax = plt.axes()
                ax.axis("off")
                ax.set_title(f"Inlier correspondences: {len(matches)} points matched")
                plot_matches(
                    ax, 
                    sfm_storage.img_pose[i].img, 
                    sfm_storage.img_pose[j].img,
                    src_keypoints[:,::-1],
                    dest_keypoints[:,::-1],
                    matches
                )
                plt.show();
    return sfm_storage


def match_sequential(sfm_storage, vis_matches, profile=True):
    """
    Match keypoints of each image pair by their descriptors
    """
    for i in range(len(sfm_storage.img_pose)-1):
        j = i+1
        # detect features and extract descriptors
        src_keypoints, src_descriptors = sfm_storage.img_pose[i].kp, sfm_storage.img_pose[i].desc
        dest_keypoints, dest_descriptors = sfm_storage.img_pose[j].kp, sfm_storage.img_pose[j].desc
        # RANSAC outlier filtering
        if profile: beg = time.time()
        robust_transform, matches = match_robust(
            src_keypoints, src_descriptors,
            dest_keypoints, dest_descriptors, 
            method='flann', 
            min_samples=4, 
            residual_threshold=100, 
            max_trials=3000, 
        )
        print(matches.shape)
        if profile: print('Match and RANSAC time:', time.time() - beg)
        # save img1-kp1-img2-kp2 matches to global helper SFM instance
        for m in matches:
            sfm_storage.img_pose[i].kp_matches[(m[0], j)] = m[1]
            sfm_storage.img_pose[j].kp_matches[(m[1], i)] = m[0]
        print(f"Feature matching: image {i} <-> image {j} ==> {len(matches)} matches")
        # vis
        FIGSIZE = (15, 10)
        if vis_matches:
            plt.figure(figsize=FIGSIZE)
            ax = plt.axes()
            ax.axis("off")
            ax.set_title(f"Inlier correspondences: {len(matches)} points matched")
            plot_matches(
                ax, 
                sfm_storage.img_pose[i].img, 
                sfm_storage.img_pose[j].img,
                src_keypoints[:,::-1],  # !!!
                dest_keypoints[:,::-1],  # !!!
                matches
            )
            plt.show();
    return sfm_storage


def detect_and_match(
    image_collection_raw,
    match_type='sequential',
    vis_matches=False
):
    image_collection = [x.astype(np.float64) / 255. for x in image_collection_raw]
    sfm_storage = SFMStorage()
    sfm_storage = extract_features(image_collection_raw, image_collection)
    if match_type == 'pairwise':
        sfm_storage = match_pairwise(sfm_storage, vis_matches)
    elif match_type == 'sequential':
        sfm_storage = match_sequential(sfm_storage, vis_matches)
    return sfm_storage
