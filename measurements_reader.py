from pathlib import Path

import numpy as np
import numpy.lib.recfunctions

import cv2

NANOSECONDS_IN_SECOND = 10**9

class MARSMeasurementsReader:
    """
    Reads measurements coming from the 'MARS logger' app.
    Currently, doesn't include images ('movie.mp4').
    """
    def __init__(self, path):
        """
        path:
            pathlib.Path or str
            A directory with 'frame_timestamps.txt', 'gyro_accel.csv', 'movie_metadata.csv', 'movie.mp4'.
        """
        self.path = Path(path)

        ##############################       Read IMU data       ##############################

        dtype = np.dtype([
            ('time', np.int64),
            ('Wx', np.float64),
            ('Wy', np.float64),
            ('Wz', np.float64),
            ('Ax', np.float64),
            ('Ay', np.float64),
            ('Az', np.float64),
        ])
        IMU_data = np.loadtxt(self.path / "gyro_accel.csv", dtype=dtype, delimiter=',', skiprows=1)

        # Time in nanoseconds
        self.timestamps_IMU = IMU_data['time']
        # Acceleration along x,y,z (in body frame) in m/s^2
        self.accelerometer  = np.lib.recfunctions.structured_to_unstructured(IMU_data[['Ax', 'Ay', 'Az']])
        # Angular velocity around x,y,z (in body frame) in rad/s
        self.gyroscope      = np.lib.recfunctions.structured_to_unstructured(IMU_data[['Wx', 'Wy', 'Wz']])

        ##############################     Read image metadata     ##############################

        dtype = np.dtype([
            ('time', np.int64),
            ('fx', np.float64),
            ('fy', np.float64),
        ])
        camera_data = np.loadtxt(self.path / "movie_metadata.csv", dtype=dtype, delimiter=',', skiprows=1, usecols=(0,1,2))

        # Time in nanoseconds (scale is consistent with `self.timestamps_IMU`, but timestamps aren't)
        self.timestamps_camera = camera_data['time']
        self.camera_focal_length = np.lib.recfunctions.structured_to_unstructured(camera_data[['fx', 'fy']])

        #############################  Check video for frame size  ##############################

        ok, sample_frame = cv2.VideoCapture(str(self.path / "movie.mp4")).read()
        assert ok
        self.DOWNSCALING = 2.0 # not used yet
        self.IMAGE_SIZE = (sample_frame.shape[1] / self.DOWNSCALING, sample_frame.shape[0] / self.DOWNSCALING)

    def get_dt_IMU(self, i):
        """
        i:
            int
            Index of the IMU measurement in `self.timestamps_IMU`.

        return:
            float
            Time in seconds between i-th and (i+1)-th IMU measurements
        """
        if i > len(self.timestamps_IMU) - 2 or i < 0:
            return 1e-16
        else:
            return (self.timestamps_IMU[i+1] - self.timestamps_IMU[i]) / NANOSECONDS_IN_SECOND

    def camera_intrinsics(self, i):
        """
        i:
            int
            Index of the camera frame in `self.timestamps_camera`.

        return:
            gtsam.Cal3DS2
        """
        return gtsam.Cal3DS2(
            *self.camera_focal_length[i],         # focal length
            0,                                    # skew
            IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2, # principal point
            0.25669783, -1.40364704,              # distortion for Huawei Honor 10: k1, k2
            -0.00332119, 0.00333635               # distortion for Huawei Honor 10: p1, p2
        )

    def get_video_reader(self):
        """
        return:
            generator, yields np.ndarray
        """
        def generator(video_reader, scaling_factor):
            while video_reader.grab():
                ok, image = video_reader.retrieve()
                assert ok
                image = cv2.resize(
                    image, (0, 0), fx=scaling_factor, fy=scaling_factor,
                    interpolation=cv2.INTER_AREA)

                yield np.rot90(image).copy()

        video_reader = cv2.VideoCapture(str(self.path / "movie.mp4"))
        return generator(video_reader, 1 / self.DOWNSCALING)
