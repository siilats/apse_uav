import sys
sys.path.append("../lib")
import unitree_arm_interface
import time
import numpy as np
from unitree import *
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--csv")

args = argParser.parse_args()

csv = args.csv

file = np.genfromtxt(csv, dtype=float, delimiter=',')
import matplotlib.pyplot as plt
plt.plot(file[:,9])
plt.show()

print("Press ctrl+\ to quit process.")

np.set_printoptions(precision=3, suppress=True)
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
ctrlComp = arm._ctrlComp
# udp = unitree_arm_interface.UDPPort(IP = "127.0.0.1", toPort=8071, ownPort=8072)
# ctrlComp.udp = udp
armModel = arm._ctrlComp.armModel

armState = unitree_arm_interface.ArmFSMState
# arm.loopOn()
# arm.setWait(True)
# arm.setFsm(armState.PASSIVE)
# arm.calibration()
# arm.loopOff()
arm.setFsmLowcmd()

duration = 1000
lastPos = arm.lowstate.getQ()
xs = []

last_joint_positions = [1.00508442885053, 1.9130590121429007, -0.9276282860951903, -0.985430726100941, -1.0050844279302438, -0.026222076561718925]
last_pos = armModel.forwardKinematics(last_joint_positions, 6)
closest_dist = 10000
closest_line = 0
middle_point = file.shape[0] / 2

# move_arm(arm, file[0, 1:7])

#read file line by line
for line in file:
    arm.q = line[1:7]
    T_forward = armModel.forwardKinematics(arm.q, 6)
    dist = last_pos[0][3] - T_forward[0][3]
    T_forward2 = armModel.forwardKinematics(arm.q + arm.qd, 6)
    speed = T_forward[0,3] -T_forward2[0,3]

    if not (line[0] > middle_point and speed < 0.01):

        if dist < closest_dist and dist > 0.05:
            closest_dist = dist
            closest_line = line[0]

    xs.append(T_forward[0,3] - T_forward2[0,3])
    print(T_forward[0:3,3])
    arm.qd = line[8:14] # set velocity
    arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6))
    arm.gripperQ = line[7]
    arm.sendRecv()# udp connection
    # print(arm.lowstate.getQ())
    time.sleep(arm._ctrlComp.dt)

arm.loopOn()
arm.startTrack(armState.JOINTCTRL)
arm.backToStart()
arm.loopOff()

#plot xs
import matplotlib.pyplot as plt
plt.plot(xs)
plt.show()

print(closest_dist)
print(closest_line)
print(file[int(closest_line)])



# for i in range(0, duration):
#     arm.q = lastPos*(1-i/duration) + targetPos*(i/duration)# set position
#     arm.qd = (targetPos-lastPos)/(duration*0.002) # set velocity
#     arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6)) # set torque
#     arm.gripperQ = -1*(i/duration)
#     arm.sendRecv()# udp connection
#     # print(arm.lowstate.getQ())
#     time.sleep(arm._ctrlComp.dt)



