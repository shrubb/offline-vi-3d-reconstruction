from collections import defaultdict

import numpy as np


class ImagePose():
    def __init__(self):
        self.img = np.empty(1)  # downsampled image used for display
        self.desc = np.empty(1)  # feature descriptor
        self.kp = []  # keypoints list in (x,y) format (not (y, x)!)

        self.T = np.zeros((4,4))  # 4x4 pose transformation matrix
        self.P = np.zeros((3,4))  # 3x4 projection matrix

        self.kp_matches = defaultdict(int)  # keypoint matches in other images
        self.kp_landmark = defaultdict(int)  # seypoint to 3d points

    # helper methods
    def kp_match_exist(self, kp_idx, img_idx): 
        return self.kp_matches[(kp_idx, img_idx)] > 0
    
    def kp_3d_exist(self, kp_idx):
        return self.kp_landmark[kp_idx] != 0

    
# 3D point
class Landmark():
    def __init__(self):  
        self.pt = np.zeros((3))  # cv::Point3f
        self.color = np.zeros((3))  # [R, G, B]
        self.seen = 0  # how many cameras have seen this point


# Helper class
class SFMStorage():
    def __init__(self):
        self.img_pose = []  # list of ImgPose class instances
        self.landmark = []  # list of Landmark class instances
        

# main storage class
class LandmarksStorage(object):
    """
    Stores all landmarks and corresponding images with keypoint coordinates.
    """
    def __init__(self, sfm_storage, min_landmark_seen):
        # self.landmarks_info[landmark_id] -> [(img_id1, (x, y)), ...]
        self.landmarks_info = defaultdict(list)
        self.sfm_storage = sfm_storage
        self.min_landmark_seen = min_landmark_seen
        self._fill_landmark_info(self.sfm_storage, self.min_landmark_seen)
        
    def _fill_landmark_info(self, sfm_storage, min_landmark_seen):
        for i in range(len(sfm_storage.img_pose)):
            curr_img_pose = SFM.img_pose[i]
            for k in range(len(curr_img_pose.kp)):
                if curr_img_pose.kp_3d_exist(k):
                    landmark_id = curr_img_pose.kp_landmark[k]
                    if sfm_storage.landmark[landmark_id].seen >= min_landmark_seen:
                        self.landmarks_info[landmark_id].append((i, curr_img_pose.kp[k]))
                    
        
#     def add_img_kp_pair(self, landmark_id, img_id, kp_coords):
#         self.landmarks_info[landmark_id].append((img_id, kp_coords))
        
    def get_num_landmarks(self):
        """
        Returns number of landmarks in the storage.
            
        return:
            int
            Number of landmarks in the storage
        """
        return len(landmarks_info)
    
    def get_landmark_info(self, landmark_id):
        """
        Returns list of image id's 
        and coordinates of corresponding keypoint on each image.
        
        landmark_id:
            int
            Landmark's id
            
        return:
            list
            List in format: [(img_id, (x, y)), ...]. 
            (x, y) are the coordinates of the keypoint in pixels (x<->cols, y<->rows)
        """
        return self.landmarks_info[landmark_id]

