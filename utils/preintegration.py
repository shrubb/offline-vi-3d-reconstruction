import numpy as np
import gtsam


def build_steps(accelerometer, gyroscope, camera_steps, imu_steps, imu_dts):
  combined_dt = []
  combined_steps = []
  combined_accel = []
  combined_gyro = []
  combined_steps.append(0)
  # combined_gyro.append(gyroscope[0])
  # combined_accel.append(accelerometer[0])
  i = 0
  if camera_steps[0] <= imu_steps[0]:
    i += 1
  for j in range(len(imu_steps)):
    if i < len(camera_steps) and imu_steps[j] < camera_steps[i] <= imu_steps[j + 1]:
      t = (camera_steps[i] - imu_steps[j]) / (imu_steps[j + 1] - imu_steps[j])
      combined_steps.append(j + i)
      combined_dt.append(imu_dts[j] * t)
      combined_dt.append(imu_dts[j] * (1 - t))
      combined_accel.append(accelerometer[j - 1] * (1 - t) + accelerometer[j] * t)
      combined_accel.append(accelerometer[j])
      combined_gyro.append(gyroscope[j - 1] * (1 - t) + gyroscope[j] * t)
      combined_gyro.append(gyroscope[j])
      i += 1
    else:
      combined_dt.append(imu_dts[j])
      combined_accel.append(accelerometer[j])
      combined_gyro.append(gyroscope[j])

  return combined_dt, combined_steps, combined_accel, combined_gyro