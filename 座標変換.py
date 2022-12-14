from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import csv
import math
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from Definition_v4 import *

file5 = 'deg30'
IMU = pd.read_csv('C:/卒論data/log_viewer2/dynamixel/' + file5 + '.csv')

IMU['Time'] = IMU['Time'] * 0.001
rest = IMU[(IMU['Time']>=0) & (IMU['Time']<=6)]
rest_acc_x1 = rest['1-AccX']
rest_acc_y1 = rest['1-AccY']
rest_acc_z1 = rest['1-AccZ']
rest_gyro_x1 = rest['1-GyroX']
rest_gyro_y1 = rest["1-GyroY"]
rest_gyro_z1 = rest["1-GyroZ"]

offset_acc_x1 = sum(rest_acc_x1)/len(rest_acc_x1)-(sum(rest_acc_x1)/len(rest_acc_x1))/(math.sqrt((sum(rest_acc_x1)/len(rest_acc_x1))**2+(sum(rest_acc_y1)/len(rest_acc_y1))**2+(sum(rest_acc_z1)/len(rest_acc_z1))**2))
offset_acc_y1 = sum(rest_acc_y1)/len(rest_acc_y1)-(sum(rest_acc_y1)/len(rest_acc_y1))/(math.sqrt((sum(rest_acc_x1)/len(rest_acc_x1))**2+(sum(rest_acc_y1)/len(rest_acc_y1))**2+(sum(rest_acc_z1)/len(rest_acc_z1))**2))
offset_acc_z1 = sum(rest_acc_z1)/len(rest_acc_z1)-(sum(rest_acc_z1)/len(rest_acc_z1))/(math.sqrt((sum(rest_acc_x1)/len(rest_acc_x1))**2+(sum(rest_acc_y1)/len(rest_acc_y1))**2+(sum(rest_acc_z1)/len(rest_acc_z1))**2))
offset_gyro_x1 = sum(rest_gyro_x1)/len(rest_gyro_x1)
offset_gyro_y1 = sum(rest_gyro_y1)/len(rest_gyro_y1)
offset_gyro_z1 = sum(rest_gyro_z1)/len(rest_gyro_z1)
print('\toffset:', offset_acc_x1, offset_acc_y1, offset_acc_z1, offset_gyro_x1, offset_gyro_y1, offset_gyro_z1)

IMU['1-AccX'] = (IMU['1-AccX'] - offset_acc_x1) * 9.80665     ## 単位変換 [m/s^2]
IMU['1-AccY'] = (IMU['1-AccY'] - offset_acc_y1) * 9.80665
IMU['1-AccZ'] = (IMU['1-AccZ'] - offset_acc_z1) * 9.80665
IMU['1-GyroX'] = (IMU['1-GyroX'] - offset_gyro_x1) * np.pi/180      ## [rad/s]
IMU['1-GyroY'] = (IMU['1-GyroY'] - offset_gyro_y1) * np.pi/180
IMU['1-GyroZ'] = (IMU['1-GyroZ'] - offset_gyro_z1) * np.pi/180

rest = IMU[(IMU['Time'] >= 0) & (IMU['Time'] <= 6)]
gravity_x1 = sum(rest['1-AccX']) / len(rest['1-AccX'])
gravity_y1 = sum(rest['1-AccY']) / len(rest['1-AccY'])
gravity_z1 = sum(rest['1-AccZ']) / len(rest['1-AccZ'])
gyro_x1 = sum(rest['1-GyroX']) / len(rest['1-GyroX'])
gyro_y1 = sum(rest['1-GyroY']) / len(rest['1-GyroY'])
gyro_z1 = sum(rest['1-GyroZ']) / len(rest['1-GyroZ'])
# print(gravity_x1,gravity_y1,gravity_z1)

## 初期姿勢角
acc = [[gravity_x1], [gravity_y1], [gravity_z1]]
roll = math.atan(acc[1][0] / acc[2][0])
pitch = math.atan(-acc[0][0] / (math.sqrt(acc[1][0] ** 2 + acc[2][0] ** 2)))
print('\t[rad]roll:', roll, 'pitch', pitch, '\n\t[deg]roll:', math.degrees(roll), 'pitch:', math.degrees(pitch))
R_acc = np.array([[-9.80665 * math.sin(pitch), 9.80665 * math.sin(roll) * math.cos(pitch), 9.80665 * math.cos(roll) * math.cos(pitch)]])
print('\tacc:', acc, '\n\tR_acc:', R_acc)        ## acc算出

gyro = [[gyro_x1], [gyro_y1], [gyro_z1]]
R_gyro = np.array([[1, math.sin(roll)*math.tan(pitch), math.cos(roll)*math.tan(pitch)], [0.0, math.cos(roll), -math.sin(roll)], [0.0, math.sin(roll)*1/math.cos(pitch), math.cos(roll)*1/math.cos(pitch)]])
angular = np.dot(R_gyro, gyro).reshape(1, -1)
print('\t姿勢角:', angular)
## 以下は，本来 Coordinate_Transformation
IMU['1-AccX'], IMU['1-AccY'], IMU['1-AccZ'] = global_Cordinate(IMU[['1-AccX', '1-AccY', '1-AccZ']], [math.degrees(roll), math.degrees(pitch), 0.0])
IMU['1-GyroX'], IMU['1-GyroY'], IMU['1-GyroZ'] = global_Cordinate(IMU[['1-GyroX', '1-GyroY', '1-GyroZ']], [math.degrees(roll), math.degrees(pitch), 0.0])
plt.plot(IMU['1-AccX'], label='X')
plt.legend()
plt.grid()
plt.show()
## 重力加速度除去
IMU["1-AccX"] = highpass(IMU["1-AccX"], 0.01, 1.8)
IMU["1-AccY"] = highpass(IMU["1-AccY"], 0.01, 1.8)
IMU["1-AccZ"] = highpass(IMU["1-AccZ"], 0.01, 1.8)
plt.plot(IMU['1-AccX'], label='X')
plt.plot(IMU['1-AccY'], label='Y')
plt.plot(IMU['1-AccZ'], label='Z')
plt.legend()
plt.grid()
plt.show()
## 姿勢角算出
IMU['1-AttX'] = attitude(IMU['Time'], IMU['1-GyroX'])
IMU['1-AttY'] = attitude(IMU['Time'], IMU['1-GyroY'])
IMU['1-AttZ'] = attitude(IMU['Time'], IMU['1-GyroZ'])
## 以下は，本来 World_Coordinate_acc
data = IMU[['1-AccX',  '1-AccY', '1-AccZ', '1-AttX', '1-AttY', '1-AttZ']]
IMU['1-AccX'], IMU['1-AccY'], IMU['1-AccZ'] = global_Coordinate_acc(data)
data = IMU[['1-GyroX', '1-GyroY', '1-GyroZ', '1-AttX', '1-AttY', '1-AttZ']]
IMU['1-GyroX'], IMU['1-GyroY'], IMU['1-GyroZ'] = global_Coordinate_acc(data)

'''
fig = plt.figure(figsize = (8,7))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.suptitle('1-Acc', fontsize=24)
fig_x = fig.add_subplot(2, 2, 1)
plt.plot(IMU["Time"], IMU['1-AccX'], "b", label="IMU1")
plt.title("X", fontsize=18)
plt.xlabel("Time[s]", fontsize=18)
plt.ylabel("Acc[m/s]", fontsize=18)
plt.tick_params(labelsize=15)
plt.legend(fontsize=15)
plt.grid()
# Acc_Y
fig_x = fig.add_subplot(2, 2, 2)
plt.plot(IMU["Time"], IMU['1-AccY'], "b", label="IMU1")
plt.title("Y", fontsize=18)
plt.xlabel("Time[s]", fontsize=15)
plt.ylabel("Acc[m/s]", fontsize=15)
plt.tick_params(labelsize=20)
plt.legend(fontsize=18)
plt.grid()
# Acc_Z
fig_x = fig.add_subplot(2, 2, 3)
plt.plot(IMU["Time"], IMU['1-AccZ'], "b", label="IMU1")
plt.title("Z", fontsize=18)
plt.xlabel("Time[s]", fontsize=18)
plt.ylabel("Acc[m/s]", fontsize=18)
plt.tick_params(labelsize=20)
plt.legend(fontsize=18)
plt.grid()

fig = plt.figure(figsize = (8,7))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.suptitle('Gyro', fontsize=24)
# Gyro_X
fig_x = fig.add_subplot(2, 2, 1)
plt.plot(IMU["Time"], IMU['1-GyroX'], "b", label="IMU1")
plt.plot(IMU["Time"], Least_Squares_Straight_Line(IMU["Time"], IMU["1-GyroX"]), "r", label="IMU1")
plt.title("X", fontsize=18)
plt.xlabel("Time[s]", fontsize=18)
plt.ylabel("Gyro[m/s]", fontsize=18)
plt.tick_params(labelsize=20)
plt.legend(fontsize=18)
plt.grid()
# # Gyro_Y
fig_x = fig.add_subplot(2, 2, 2)
plt.plot(IMU["Time"], IMU['1-GyroY'], "b", label="IMU1")
plt.plot(IMU["Time"], Least_Squares_Straight_Line(IMU["Time"], IMU["1-GyroY"]) - Least_Squares_Straight_Line(IMU["Time"], IMU["1-GyroY"]).iloc[0], "r", label="IMU1")
plt.title("Y", fontsize=18)
plt.xlabel("Time[s]", fontsize=18)
plt.ylabel("Gyro[m/s]", fontsize=18)
plt.tick_params(labelsize=20)
plt.legend(fontsize=18)
plt.grid()
# # Gyro_Z
fig_x = fig.add_subplot(2, 2, 3)
plt.plot(IMU["Time"], IMU['1-GyroZ'], "b", label="IMU1")
plt.plot(IMU["Time"], Least_Squares_Straight_Line(IMU["Time"], IMU["1-GyroZ"]), "r", label="IMU1")
plt.title("Z", fontsize=18)
plt.xlabel("Time[s]", fontsize=18)
plt.ylabel("Gyro[m/s]", fontsize=18)
plt.tick_params(labelsize=20)
plt.legend(fontsize=18)
plt.grid()
print(IMU['1-AccX'].head())
fig = plt.figure(figsize = (8,7))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.suptitle('1-Att', fontsize=24)
# Att_X
fig_x = fig.add_subplot(2, 2, 1)
plt.plot(IMU["Time"], IMU['1-AttX'], "b", label="IMU1")
plt.title("X", fontsize=18)
plt.xlabel("Time[s]", fontsize=18)
plt.ylabel("angle[m/s]", fontsize=18)
plt.tick_params(labelsize=20)
plt.legend(fontsize=18)
plt.grid()
# Att_Y
fig_x = fig.add_subplot(2, 2, 2)
plt.plot(IMU["Time"], IMU['1-AttY'], "b", label="IMU1")
plt.title("Y", fontsize=18)
plt.xlabel("Time[s]", fontsize=18)
plt.ylabel("angle[m/s]", fontsize=18)
plt.tick_params(labelsize=20)
plt.legend(fontsize=18)
plt.grid()
# Att_Z
fig_x = fig.add_subplot(2, 2, 3)
plt.plot(IMU["Time"], IMU['1-AttZ'], "b", label="IMU1")
plt.title("Z", fontsize=18)
plt.xlabel("Time[s]", fontsize=18)
plt.ylabel("angle[m/s]", fontsize=18)
plt.tick_params(labelsize=20)
plt.legend(fontsize=18)
plt.grid()
plt.show()
'''
IMU['1-VeloX'] = velocity(IMU['Time'], IMU['1-AccX'], 'X')
# IMU['1-VeloY'] = velocity(IMU['Time'], IMU['1-AccY'], 'Y')
# IMU['1-VeloZ'] = velocity(IMU['Time'], IMU['1-AccZ'], 'Z')
plt.suptitle('Velocity', fontsize=24)
plt.legend()
plt.show()
IMU['1-PosX'] = position(IMU['Time'], IMU['1-VeloX'], 'X')
# IMU['1-PosY'] = position(IMU['Time'], IMU['1-VeloY'], 'Y')
# IMU['1-PosZ'] = position(IMU['Time'], IMU['1-VeloZ'], 'Z')
plt.suptitle('Position', fontsize=24)
plt.legend()
plt.show()
