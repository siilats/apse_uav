import sys
sys.path.append("../lib")
import unitree_arm_interface
import time
import numpy as np

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--csv")

args = argParser.parse_args()

csv = args.csv

file = np.genfromtxt(csv, dtype=float, delimiter=',')

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

#read file line by line
for line in file:
    arm.q = line[1:7]
    T_forward = armModel.forwardKinematics(arm.q, 6)
    T_forward2 = armModel.forwardKinematics(arm.q + arm.qd, 6)
    xs.append(T_forward[0,3] - T_forward2[0,3])
    print(T_forward[0:3,3])
    arm.qd = line[8:14] # set velocity
    arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6))
    arm.gripperQ = line[7]
    arm.sendRecv()# udp connection
    # print(arm.lowstate.getQ())
    time.sleep(arm._ctrlComp.dt)

#plot xs
import matplotlib.pyplot as plt
plt.plot(xs)
plt.show()


# for i in range(0, duration):
#     arm.q = lastPos*(1-i/duration) + targetPos*(i/duration)# set position
#     arm.qd = (targetPos-lastPos)/(duration*0.002) # set velocity
#     arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6)) # set torque
#     arm.gripperQ = -1*(i/duration)
#     arm.sendRecv()# udp connection
#     # print(arm.lowstate.getQ())
#     time.sleep(arm._ctrlComp.dt)

# arm.loopOn()
# arm.backToStart()
# arm.loopOff()

