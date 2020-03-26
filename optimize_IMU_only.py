import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import gtsam
from gtsam.utils.plot import plot_pose3

if __name__ == '__main__':
    imu_measurements, _, _, true_poses = np.load('circle_gold.npy')

    ##############################   Algorithm parameters   ############################
    import argparse
    params = argparse.Namespace()

    # Time step length.
    # In real world, it will be different at each measurement,
    # and you'll have to take dtᵢ from data.
    params.dt = 1e-2
    
    # IMU bias
    params.accelerometer_bias = np.array([0, 0.1, 0])
    params.gyroscope_bias = np.array([0, 0, 0])
    params.IMU_bias = gtsam.imuBias_ConstantBias(params.accelerometer_bias, params.gyroscope_bias)
    
    # 'circle_gold.npy' simulates a loop with forward velocity 2 m/s,
    # while pitching up with angular velocity 30 degrees/sec.
    params.initial_velocity = np.array([2, 0, 0])
    params.initial_angular_vel = np.array([0, -math.radians(30), 0]) # not used here
    params.initial_pose = gtsam.Pose3()
    params.initial_state = gtsam.NavState(params.initial_pose, params.initial_velocity)

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
    current_preintegrated_IMU = gtsam.PreintegratedImuMeasurements(params.preintegration_params, params.IMU_bias)

    # The certainty (covariance) of the initial position estimate
    # "Isotropic" means diagonal with equal sigmas
    params.initial_pose_covariance = gtsam.noiseModel_Isotropic.Sigma(6, 0.1)
    params.initial_velocity_covariance = gtsam.noiseModel_Isotropic.Sigma(3, 0.1)

    ###############################    Build the factor graph   ####################################
    
    factor_graph = gtsam.NonlinearFactorGraph()
    # A technical hack for defining variable names in GTSAM python bindings
    def symbol(letter, index): return int(gtsam.symbol(ord(letter), index))

    # Add a prior factor on the initial position
    factor_graph.push_back(gtsam.PriorFactorPose3(symbol('x', 0), params.initial_pose, params.initial_pose_covariance))
    factor_graph.push_back(gtsam.PriorFactorVector(symbol('v', 0), params.initial_velocity, params.initial_velocity_covariance))

    # Add IMU factors (or "motion model"/"transition" factors).
    # Ideally, we would add factors between every pair (xᵢ₋₁, xᵢ). But, to save computations,
    # we will add factors between pairs (x₀, xₖ), (xₖ, x₂ₖ) etc., and as an IMU "measurement"
    # between e.g. x₀ and xₖ we will use combined (pre-integrated) measurements `0, 1, ..., k-1`.
    # Below, `k == PREINTEGRATE_EVERY_STEPS`.
    PREINTEGRATE_EVERY_STEPS = 25

    # For code generalization, create pairs (0, k), (k, 2k), (2k, 3k), ..., (mk, N-1)
    preintegration_steps = list(range(0, len(imu_measurements), PREINTEGRATE_EVERY_STEPS))
    if preintegration_steps[-1] != len(imu_measurements) - 1: # don't miss the last measurements
        preintegration_steps.append(len(imu_measurements) - 1)
    # An iterator over those pairs
    imu_factor_pairs = zip(preintegration_steps[:-1], preintegration_steps[1:])
    current_imu_factor_pair = next(imu_factor_pairs)

    # Clear the accumulated value
    current_preintegrated_IMU.resetIntegration()

    for i, imu_measurement in enumerate(imu_measurements):
        measured_acceleration, measured_angular_vel = imu_measurement[:3], imu_measurement[3:]

        if i == current_imu_factor_pair[1]:
            # Add IMU factor
            factor = gtsam.ImuFactor(
                symbol('x', current_imu_factor_pair[0]),
                symbol('v', current_imu_factor_pair[0]),
                symbol('x', current_imu_factor_pair[1]),
                symbol('v', current_imu_factor_pair[1]),
                symbol('b', 0),
                current_preintegrated_IMU)
            factor_graph.push_back(factor)
            
            # Start accumulating from scratch
            current_preintegrated_IMU.resetIntegration()

            # Get the next pair of indices
            try:
                current_imu_factor_pair = next(imu_factor_pairs)
            except StopIteration:
                assert i == len(imu_measurements) - 1

        # Accumulate the current measurement
        current_preintegrated_IMU.integrateMeasurement(measured_acceleration, measured_angular_vel, params.dt)

    ############################# Specify initial values for optimization #################################

    # The optimization will start from these initial values of our target variables:
    initial_values = gtsam.Values()

    # The initial value for IMU bias
    initial_values.insert(symbol('b', 0), params.IMU_bias)


    # The initial values for coordinates (x) and velocities (v), estimated by IMU preintegration.
    # Note that our variables are only for those steps that have been chosen into `preintegration_steps`.
    preintegration_steps_set = set(preintegration_steps)
    # Clear the accumulated value from the previous code section
    current_preintegrated_IMU.resetIntegration()

    for i, imu_measurement in enumerate(imu_measurements):
        if i in preintegration_steps_set:
            predicted_nav_state = current_preintegrated_IMU.predict(params.initial_state, params.IMU_bias)
            initial_values.insert(symbol('x', i), predicted_nav_state.pose())
            initial_values.insert(symbol('v', i), predicted_nav_state.velocity())

        current_preintegrated_IMU.integrateMeasurement(imu_measurements[i, :3], imu_measurements[i, 3:], params.dt)

    ###############################    Optimize the factor graph   ####################################

    # Use the Levenberg-Marquardt algorithm
    optimization_params = gtsam.LevenbergMarquardtParams()
    optimization_params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, initial_values, optimization_params)
    optimization_result = optimizer.optimize()

    ###############################        Plot the solution       ####################################

    for i in preintegration_steps:
        # Ground truth pose
        plot_pose3(1, gtsam.Pose3(true_poses[i]), 0.3)
        # Estimated pose
        plot_pose3(1, optimization_result.atPose3(symbol('x', i)), 0.1)

    ax = plt.gca()
    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.figure(1).suptitle("Large poses: ground truth, small poses: estimate")

    plt.ioff()
    plt.show()
