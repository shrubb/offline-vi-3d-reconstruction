import numpy as np

def build_steps(accelerometer, gyroscope, camera_steps, imu_steps, imu_dts):
  combined_dt = []
  combined_steps = []
  combined_accel = []
  combined_gyro = []

  # i: the upcoming camera measurement
  # j: currently processed original IMU measurement
  # k: last (exclusive) new IMU measurement index, i.e. `len(combined_steps)`
  i = 0
  k = 0

  for j in range(len(imu_steps)-1):
    if i < len(camera_steps) and imu_steps[j] <= camera_steps[i] <= imu_steps[j+1]:
      # Split this IMU interval into two to "insert" the video frame point

      # Interpolate the measurements linearly
      a = (camera_steps[i] - imu_steps[j]) / (imu_steps[j+1] - imu_steps[j])
      combined_dt.append(imu_dts[j] * a)
      combined_dt.append(imu_dts[j] * (1 - a))
      combined_accel.append(accelerometer[j])
      combined_accel.append(accelerometer[j] * a + accelerometer[j+1] * (1 - a))
      combined_gyro.append(gyroscope[j])
      combined_gyro.append(gyroscope[j] * a + gyroscope[j+1] * (1 - a))
      combined_steps.append(k+1)
      k += 2
      i += 1
    else:
      # Simply append the current IMU measurement
      combined_dt.append(imu_dts[j])
      combined_accel.append(accelerometer[j])
      combined_gyro.append(gyroscope[j])
      k += 1

  if i != len(camera_steps):
    raise RuntimeError("Run out of IMU measurements but still have video frames")

  return combined_dt, combined_steps, combined_accel, combined_gyro