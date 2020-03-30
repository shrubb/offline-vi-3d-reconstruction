import gtsam
from gtsam.utils.plot import plot_pose3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from measurements_reader import MARSMeasurementsReader

import utils.geometry
import utils.preintegration
import numpy as np
import math

if __name__ == '__main__':
    measurements = MARSMeasurementsReader("sample_data/closed_trajectory")

    ##############################   Algorithm parameters   ############################

    import argparse
    params = argparse.Namespace()

    # IMU bias
    params.accelerometer_bias = np.array([0, 0, 0])
    params.gyroscope_bias = np.array([0, 0, 0])
    params.IMU_bias = gtsam.imuBias_ConstantBias(params.accelerometer_bias, params.gyroscope_bias)
    params.IMU_bias_covariance = gtsam.noiseModel_Isotropic.Sigma(6, 0.2)

    # Assume initial velocity is zero (consider dropping this assumption when the visual part comes!)
    initial_velocity = np.array([0, 0, 0])
    # Assume w.l.o.g. that we start from (0, 0, 0)
    initial_position = gtsam.Point3()
    # Assuming initial acceleration is zero, estimate the initial orientation
    initial_orientation = utils.geometry.estimate_initial_orientation(measurements.accelerometer[0])

    initial_pose = gtsam.Pose3(initial_orientation, initial_position)
    params.initial_state = gtsam.NavState(initial_pose, initial_velocity)

    # IMU preintegration algorithm parameters
    # "U" means "Z axis points up"; "MakeSharedD" would mean "Z axis points along the gravity"
    preintegration_params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)  # 9.81 is the gravity force
    # Realistic noise parameters
    kGyroSigma = math.radians(0.5) / 60  # 0.5 degree ARW
    kAccelSigma = 0.1 / 60  # 10 cm VRW
    preintegration_params.setGyroscopeCovariance(kGyroSigma ** 2 * np.identity(3, np.float))
    preintegration_params.setAccelerometerCovariance(kAccelSigma ** 2 * np.identity(3, np.float))
    preintegration_params.setIntegrationCovariance(0.0000001 ** 2 * np.identity(3, np.float))
    params.preintegration_params = preintegration_params

    # The stateful class that is responsible for preintegration
    current_preintegrated_IMU = gtsam.PreintegratedCombinedMeasurements(params.preintegration_params, params.IMU_bias)

    # The certainty (covariance) of the initial position estimate
    # "Isotropic" means diagonal with equal sigmas
    params.initial_pose_covariance = gtsam.noiseModel_Isotropic.Sigma(6, 0.1)
    params.initial_velocity_covariance = gtsam.noiseModel_Isotropic.Sigma(3, 0.1)

    ###############################    Build the factor graph   ####################################

    factor_graph = gtsam.NonlinearFactorGraph()

    # A technical hack for defining variable names in GTSAM python bindings
    def symbol(letter, index):
        return int(gtsam.symbol(ord(letter), index))

    imu_dts = [measurements.get_dt_IMU(i) for i in range(len(measurements.timestamps_IMU))]
    dt, preintegration_steps, acc, gyr = utils.preintegration.build_steps(
        measurements.accelerometer, measurements.gyroscope, measurements.timestamps_camera,
        measurements.timestamps_IMU, imu_dts)

    # Add IMU factors (or "motion model"/"transition" factors).
    # Ideally, we would add factors between every pair (xᵢ₋₁, xᵢ). But, to save computations,
    # we will add factors between pairs (x₀, xₖ), (xₖ, x₂ₖ) etc., and as an IMU "measurement"
    # between e.g. x₀ and xₖ we will use combined (pre-integrated) measurements `0, 1, ..., k-1`.
    # Below, `k == PREINTEGRATE_EVERY_FRAMES`.
    PREINTEGRATE_EVERY_FRAMES = 6
    preintegration_steps = preintegration_steps[::PREINTEGRATE_EVERY_FRAMES]

    # For code generalization, create pairs (0, k), (k, 2k), (2k, 3k), ..., (mk, N-1)
    # An iterator over those pairs
    imu_factor_pairs = zip(preintegration_steps[:-1], preintegration_steps[1:])
    current_imu_factor_pair = next(imu_factor_pairs)
    # Clear the accumulated value
    current_preintegrated_IMU.resetIntegration()

    for i, (measured_acceleration, measured_angular_vel) in enumerate(zip(acc, gyr)):
        if i == current_imu_factor_pair[1]:
            # Add IMU factor
            factor = gtsam.CombinedImuFactor(
                symbol('x', current_imu_factor_pair[0]),
                symbol('v', current_imu_factor_pair[0]),
                symbol('x', current_imu_factor_pair[1]),
                symbol('v', current_imu_factor_pair[1]),
                symbol('b', current_imu_factor_pair[0]),
                symbol('b', current_imu_factor_pair[1]),
                current_preintegrated_IMU)
            factor_graph.push_back(factor)

            # Start accumulating from scratch
            current_preintegrated_IMU.resetIntegration()

            # Get the next pair of indices
            try:
                current_imu_factor_pair = next(imu_factor_pairs)
            except StopIteration:
                pass

        # Accumulate the current measurement
        current_preintegrated_IMU.integrateMeasurement(measured_acceleration, measured_angular_vel, dt[i])

    # Add a prior factor on the initial state
    factor_graph.push_back(
        gtsam.PriorFactorPose3(symbol('x', preintegration_steps[0]), params.initial_state.pose(), params.initial_pose_covariance))
    factor_graph.push_back(
        gtsam.PriorFactorVector(symbol('v', preintegration_steps[0]), params.initial_state.velocity(), params.initial_velocity_covariance))
    # factor_graph.push_back(
    #     gtsam.PriorFactorConstantBias(symbol('b', preintegration_steps[0]), params.IMU_bias, params.IMU_bias_covariance))

    # Other example priors, e.g. the intial and the final states should be roughly identical:
    factor_graph.push_back(
        gtsam.PriorFactorPose3(symbol('x', preintegration_steps[-1]), params.initial_state.pose(), params.initial_pose_covariance))
    factor_graph.push_back(
        gtsam.PriorFactorVector(symbol('v', preintegration_steps[-1]), params.initial_state.velocity(), params.initial_velocity_covariance))

    ############################# Specify initial values for optimization #################################

    # The optimization will start from these initial values of our target variables:
    initial_values = gtsam.Values()

    # The initial values for coordinates (x) and velocities (v), estimated by IMU preintegration.
    # Note that our variables are only for those steps that have been chosen into `preintegration_steps`.
    preintegration_steps_set = set(preintegration_steps)
    # Clear the accumulated value from the previous code section
    current_preintegrated_IMU.resetIntegration()

    for i, (measured_acceleration, measured_angular_vel) in enumerate(zip(acc, gyr)):
        if i in preintegration_steps_set:
            predicted_nav_state = current_preintegrated_IMU.predict(params.initial_state, params.IMU_bias)
            initial_values.insert(symbol('x', i), predicted_nav_state.pose())
            initial_values.insert(symbol('v', i), predicted_nav_state.velocity())
            initial_values.insert(symbol('b', i), params.IMU_bias)

        current_preintegrated_IMU.integrateMeasurement(measured_acceleration, measured_angular_vel, dt[i])

    ###############################    Optimize the factor graph   ####################################

    # Use the Levenberg-Marquardt algorithm
    optimization_params = gtsam.LevenbergMarquardtParams()
    optimization_params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, initial_values, optimization_params)
    optimization_result = optimizer.optimize()

    ###############################        Plot the solution       ####################################

    figure = plt.figure(1)
    axes = plt.gca(projection='3d')
    axes.set_xlim3d(-2, 2)
    axes.set_ylim3d(-2, 2)
    axes.set_zlim3d(-2, 2)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.margins(0)
    figure.suptitle("Large/small poses: initial/optimized estimate")

    for i in preintegration_steps:
        # Initial estimate from IMU preintegration
        plot_pose3(1, initial_values.atPose3(symbol('x', i)), 0.2)
        # Optimized estimate
        plot_pose3(1, optimization_result.atPose3(symbol('x', i)), 0.08)

        plt.pause(0.01)

    predicted_imu_biases = [optimization_result.atimuBias_ConstantBias(symbol('b', i)) for i in preintegration_steps]
    predicted_acccelerometer_biases = np.stack([x.accelerometer() for x in predicted_imu_biases])
    predicted_gyroscope_biases      = np.stack([x.gyroscope()     for x in predicted_imu_biases])

    plt.figure(2)
    for parameter in predicted_acccelerometer_biases.T:
        plt.plot(parameter)
    for parameter in predicted_gyroscope_biases.T:
        plt.plot(parameter)
    plt.grid()
    plt.title("Predicted accelerometer and gyroscope biases")
    plt.legend(['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])

    plt.ioff()
    plt.show()
