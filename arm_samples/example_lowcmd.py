import sys
sys.path.append("../lib")
import unitree_arm_interface
import time
import numpy as np

print("Press ctrl+\ to quit process.")

np.set_printoptions(precision=3, suppress=True)
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
ctrlComp = arm._ctrlComp
# udp = unitree_arm_interface.UDPPort(IP = "127.0.0.1", toPort=8071, ownPort=8072)
# ctrlComp.udp = udp
armModel = arm._ctrlComp.armModel

armState = unitree_arm_interface.ArmFSMState
arm.loopOn()
arm.setWait(True)
arm.setFsm(armState.PASSIVE)
arm.calibration()
arm.loopOff()
arm.setFsmLowcmd()

duration = 1000
lastPos = arm.lowstate.getQ()
targetPos = np.array([0.0, 1, -1, -0.24, 0.0, 0.0]) #forward

for i in range(0, duration):
    arm.q = lastPos*(1-i/duration) + targetPos*(i/duration)# set position
    arm.qd = (targetPos-lastPos)/(duration*0.002) # set velocity
    arm.tau = armModel.inverseDynamics(
        arm.q, arm.qd, np.zeros(6), np.zeros(6)) / 1000# set torque
    # arm.gripperQ = -1*(i/duration)
    arm.sendRecv()# udp connection
    # print(arm.lowstate.getQ())
    time.sleep(arm._ctrlComp.dt)

targetPos = arm.q
lastPos = arm.q
print("middle point")

for i in range(0, duration):
    arm.q = lastPos*(1-i/duration) + targetPos*(i/duration)# set position
    arm.qd = (targetPos-lastPos)/(duration*0.002) # set velocity
    arm.tau = armModel.inverseDynamics(
        arm.q, arm.qd, np.zeros(6), np.zeros(6)) / 1000 # set torque
    # arm.gripperQ = -1*(i/duration)
    arm.sendRecv()# udp connection
    # print(arm.lowstate.getQ())
    time.sleep(arm._ctrlComp.dt)

arm.loopOn()
arm.backToStart()
arm.loopOff()