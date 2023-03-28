# Make sure to have CoppeliaSim running, with followig scene loaded:
#
# scenes/messaging/ikMovementViaRemoteApi.ttt
#
# Do not launch simulation, then run this script

from zmqRemoteApi import RemoteAPIClient
import time
import numpy as np
import ctypes

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

sim.stopSimulation()
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

executedMovId = 'notReady'
targetArm = '/z1_robot'
stringSignalName = targetArm + '_executedMovId'
robotBaseHandle = sim.getObject(targetArm)
scriptHandle = sim.getScript(sim.scripttype_childscript, robotBaseHandle)

tipHandle = sim.getObject('/tip')
targetHandle = sim.getObject('/target')
yokeBoardHandle = sim.getObject('/yokeBoard')
tipArucoHandle = sim.getObject('/middle_tip_marker')


def waitForMovementExecuted(id_):
    global executedMovId, stringSignalName
    while executedMovId != id_:
        s = sim.getStringSignal(stringSignalName)
        executedMovId = s


def waitForMovementExecutedAsync(id_):
    global executedMovId
    while executedMovId != id_:
        s = sim.waitForSignal(id_)
        if s is True:
            executedMovId = id_


# Set-up some movement variables:
maxVel = 0.1
maxAccel = 0.01



# Start simulation:
sim.startSimulation()

# Wait until ready:
waitForMovementExecutedAsync('z1_ready')

#this part is illustrated in labeled_image_sample_3.png
tip_aruco_rb_orig = sim.getObjectPose(tipArucoHandle, robotBaseHandle)
tip_aruco_rb = [+4.7920e-01, 0, +5.5858e-01, -0.5, 0.5, -0.5, 0.5]
tip_aruco_w = sim.getObjectPose(tipArucoHandle, -1)
robot_base_w_orig = sim.getObjectPose(robotBaseHandle, -1)

tip_aruco_rb_orig_invert = sim.callScriptFunction('remoteApi_invertPose', scriptHandle, tip_aruco_rb_orig)
robot_base_w = sim.multiplyPoses(tip_aruco_w, tip_aruco_rb_orig_invert)

yoke_aruco_rb_orig = sim.getObjectPose(yokeBoardHandle, robotBaseHandle)
yoke_aruco_w = sim.getObjectPose(yokeBoardHandle, -1)
robot_base_w_invert = sim.callScriptFunction('remoteApi_invertPose', scriptHandle, robot_base_w)
yoke_aruco_rb = sim.multiplyPoses(robot_base_w_invert, yoke_aruco_w)


# Get initial pose:
initialPose, initialConfig = sim.callScriptFunction('remoteApi_getPoseAndConfig_base', scriptHandle)
zeroPose = [0.5, 0, 0.5, 0, 0, 0, 1]
# Send first movement sequence:

targetPose = yoke_aruco_rb

movementData = {
    'id': 'movSeq1',
    'type': 'mov',
    'targetPose': targetPose,
    'maxVel': maxVel,
    'maxAccel': maxAccel
}

sim.callScriptFunction('remoteApi_movementDataFunction',scriptHandle,movementData)
# Execute first movement sequence:
sim.callScriptFunction('remoteApi_executeMovement', scriptHandle, 'movSeq1')

# Wait until above movement sequence finished executing:
waitForMovementExecutedAsync('movSeq1')

# Send second and third movement sequence, where third one should execute
# immediately after the second one:
# targetPose = yoke_rb
#
# movementData = {
#     'id': 'movSeq2',
#     'type': 'mov',
#     'targetPose': targetPose,
#     'maxVel': maxVel,
#     'maxAccel': maxAccel
# }
# sim.callScriptFunction('remoteApi_movementDataFunction',scriptHandle,movementData)
# movementData = {
#     'id': 'movSeq3',
#     'type': 'mov',
#     'targetPose': initialPose,
#     'maxVel': maxVel,
#     'maxAccel': maxAccel
# }
# sim.callScriptFunction('remoteApi_movementDataFunction',scriptHandle,movementData)
#
# # Execute second and third movement sequence:
# sim.callScriptFunction('remoteApi_executeMovement',scriptHandle,'movSeq2')
# # sim.callScriptFunction('remoteApi_executeMovement',scriptHandle,'movSeq3')
#
# # Wait until above 2 movement sequences finished executing:
# waitForMovementExecutedAsync('movSeq3')

# sim.stopSimulation()

print('Program ended')
