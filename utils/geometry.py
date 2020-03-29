import numpy as np
import gtsam

def get_rotation_from_axis(axis, angle):
    """
    Compute the matrix that rotates around `axis` by `angle` radians.
    See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle.

    axis:
        np.ndarray, shape == (3,)
    angle:
        float
    
    return:
        gtsam.Rot3
    """
    axis = gtsam.Unit3(gtsam.Point3(axis))
    axis_np = axis.point3().vector()
    return gtsam.Rot3(
        np.cos(angle) * np.eye(3) + \
        np.sin(angle) * axis.skew() + \
        (1 - np.cos(angle)) * np.outer(axis_np, axis_np)
    )

def estimate_initial_orientation(initial_acceleration):
    """
    Compute an orientation in which the input measurement `initial_acceleration` is
    consistent with (1) zero acceleration and (2) gravity along the global negative Z-axis.

    initial_acceleration:
        np.ndarray, shape == (3,)

    return:
        gtsam.Rot3
    """
    # TODO: will fail if the actual initial orientation is strictly upside down

    # Normalized gravity
    desired_acceleration = np.array([0, 0, 1])
    # Normalized measurement
    initial_acceleration = initial_acceleration / np.linalg.norm(initial_acceleration)
    # Their average which to rotate around
    rotation_axis = initial_acceleration + desired_acceleration

    return get_rotation_from_axis(rotation_axis, np.pi)
