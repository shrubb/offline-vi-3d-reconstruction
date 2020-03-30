import gtsam

from pathlib import Path

import numpy as np
import numpy.lib.recfunctions

import cv2

NANOSECONDS_IN_SECOND = 10**9

class MARSMeasurementsReader:
    """
    Reads measurements coming from the 'MARS logger' app.
    """
    def __init__(self, path):
        """
        path:
            pathlib.Path or str
            A directory with 'frame_timestamps.txt', 'gyro_accel.csv', 'movie_metadata.csv', 'movie.mp4'.
        """
        self._downscaling = 2.0
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
        self.camera_focal_length /= self._downscaling

        for self._skip_first_n_frames, timestamp_camera in enumerate(self.timestamps_camera):
            if timestamp_camera >= self.timestamps_IMU[0]:
                break
        if self.timestamps_camera[-1] > self.timestamps_IMU[-1]:
            raise NotImplementedError("IMU measurements end before video frames do")

        self.timestamps_camera = self.timestamps_camera[self._skip_first_n_frames:]
        self.camera_focal_length = self.camera_focal_length[self._skip_first_n_frames:]

        #############################  Check video for frame size  ##############################

        ok, sample_frame = cv2.VideoCapture(str(self.path / "movie.mp4")).read()
        assert ok
        self._image_size = (sample_frame.shape[1] / self._downscaling, sample_frame.shape[0] / self._downscaling)

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
            self._image_size[0] / 2,              # principal point
            self._image_size[1] / 2,
            0.25669783, -1.40364704,              # distortion for Huawei Honor 10: k1, k2
            -0.00332119, 0.00333635               # distortion for Huawei Honor 10: p1, p2
        )

    def get_video_reader(self, steps=None):
        """
        steps:
            iterable of int or None
            Indices of frames which to yield.

        return:
            generator, yields np.ndarray
        """
        def generator(video_reader, scaling_factor, steps):
            steps = sorted(steps, reverse=True)
            frame_idx = 0

            while video_reader.grab():
                if len(steps) > 0 and frame_idx == steps[-1]:
                    steps.pop()

                    ok, image = video_reader.retrieve()
                    assert ok
                    image = cv2.resize(image, (0, 0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

                    yield np.rot90(image).copy()

                frame_idx += 1

        video_reader = cv2.VideoCapture(str(self.path / "movie.mp4"))
        # Skip the first frame because it's not represented in 'movie_metadata.csv' by MARS;
        # Maybe skip more frames because they could start earlier than IMU
        for _ in range(self._skip_first_n_frames + 1):
            video_reader.grab()

        if steps is None:
            steps = range(int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)))

        return generator(video_reader, 1 / self._downscaling, steps)
