# Adapted from GTSAM's examples/PreintegrationExample.py
import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import gtsam
from gtsam.utils.plot import plot_pose3

POSES_FIG = 1

def plotGroundTruthPose(i):
    # plot ground truth pose, as well as prediction from integrated IMU measurements
    actualPose = gtsam.Pose3(true_poses[i])
    plot_pose3(POSES_FIG, actualPose, 0.3)

    ax = plt.gca()
    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)

    plt.pause(0.01)

if __name__ == '__main__':
    imu_measurements, predicted_poses_gold, predicted_vNav_gold, true_poses = np.load('circle_gold.npy')

    #################   Algorithm parameters   #################
    import argparse
    params = argparse.Namespace()

    # Time step length
    params.dt = 1e-2
    
    # IMU bias
    params.accelerometer_bias = np.array([0, 0.1, 0])
    params.gyroscope_bias = np.array([0, 0, 0])
    params.IMU_bias = gtsam.imuBias_ConstantBias(params.accelerometer_bias, params.gyroscope_bias)
    
    # 'circle_gold.npy' simulates a loop with forward velocity 2 m/s,
    # while pitching up with angular velocity 30 degrees/sec
    params.initial_v = np.array([2, 0, 0])
    params.initial_w = np.array([0, -math.radians(30), 0])
    params.initial_pose = gtsam.Pose3()
    params.initial_state = gtsam.NavState(params.initial_pose, params.initial_v)

    # IMU preintegration algorithm parameters
    # "U" means "Z axis points up"; "MakeSharedD" would mean "Z axis points along the gravity" 
    preintegration_params = gtsam.PreintegrationParams.MakeSharedU(10) # 10 is the gravity force
    # Realistic noise parameters
    kGyroSigma = math.radians(0.5) / 60  # 0.5 degree ARW
    kAccelSigma = 0.1 / 60  # 10 cm VRW
    preintegration_params.setGyroscopeCovariance(kGyroSigma ** 2 * np.identity(3, np.float))
    preintegration_params.setAccelerometerCovariance(kAccelSigma ** 2 * np.identity(3, np.float))
    preintegration_params.setIntegrationCovariance(0.0000001 ** 2 * np.identity(3, np.float))
    params.preintegration_params = preintegration_params

    # The stateful class that is responsible for preintegration
    currentPreIntegratedIMU = gtsam.PreintegratedImuMeasurements(params.preintegration_params, params.IMU_bias)

    PREDICTION_FREQ = 25

    for i, imu_measurement in enumerate(imu_measurements):
        if i % PREDICTION_FREQ == 0:
            plotGroundTruthPose(i)
            predictedNavState = currentPreIntegratedIMU.predict(params.initial_state, params.IMU_bias)

            plot_pose3(POSES_FIG, predictedNavState.pose(), 0.1)

            assert np.allclose(predictedNavState.pose().matrix(), predicted_poses_gold[i])
            assert np.allclose(predictedNavState.velocity(), predicted_vNav_gold[i])

        measured_accuracy, measured_angular_vel = imu_measurement[:3], imu_measurement[3:]
        currentPreIntegratedIMU.integrateMeasurement(measured_accuracy, measured_angular_vel, params.dt)

    plt.ioff()
    plt.show()
