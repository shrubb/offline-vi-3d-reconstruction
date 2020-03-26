from pathlib import Path
import numpy as np

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
        path = Path(path)

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
        IMU_data = np.loadtxt(path / "gyro_accel.csv", dtype=dtype, delimiter=',', skiprows=1)

        # Time in nanoseconds
        self.timestamps_IMU = IMU_data['time']
        # Acceleration along x,y,z (in body frame) in m/s^2
        self.accelerometer  = IMU_data[['Ax', 'Ay', 'Az']].view(np.float64).reshape(len(IMU_data), 3)
        # Angular velocity around x,y,z (in body frame) in rad/s
        self.gyroscope      = IMU_data[['Wx', 'Wy', 'Wz']].view(np.float64).reshape(len(IMU_data), 3)

        ##############################     Read image metadata     ##############################

        dtype = np.dtype([
            ('time', np.int64),
            ('fx', np.float64),
            ('fy', np.float64),
            ('focal_length', np.float64),
            ('focus_distance', np.float64),
        ])
        camera_data = np.loadtxt(path / "movie_metadata.csv", dtype=dtype, delimiter=',', skiprows=1, usecols=(0,1,2,8,9))

        # Time in nanoseconds (consistent with `self.timestamps_IMU`)
        self.timestamps_camera = camera_data['time']

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
