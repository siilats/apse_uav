# Make sure to have CoppeliaSim running, with followig scene loaded:
#
# scenes/messaging/ikMovementViaRemoteApi.ttt
#
# Do not launch simulation, then run this script

from zmqRemoteApi import RemoteAPIClient
import time
import numpy as np

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

# base = [0, -0.65, 0.25, 0, 0, 0, 0]
# yoke = [0.275, -0.255, 0.6, 0, 0, 0, 0]
# Send movement sequences:
tip_rb = sim.getObjectPose(tipHandle, robotBaseHandle)
yoke_rb = sim.getObjectPose(yokeBoardHandle, robotBaseHandle)
yoke_rb_matrix = sim.poseToMatrix(yoke_rb)
yoke_rb_euler = sim.getEulerAnglesFromMatrix(yoke_rb_matrix)

print('yoke_rb_euler xyz: ', yoke_rb_euler[0], yoke_rb_euler[1] - np.pi / 2, yoke_rb_euler[2] + np.pi / 2)
print('yoke_rb_xyz: ', yoke_rb[0], yoke_rb[1], yoke_rb[2])

# Start simulation:
sim.startSimulation()


# Wait until ready:
waitForMovementExecutedAsync('z1_ready')

# Get initial pose:
initialPose, initialConfig = sim.callScriptFunction('remoteApi_getPoseAndConfig_base',scriptHandle)
zeroPose = [0.5, 0, 0.5, 0, 0, 0, 1]
# Send first movement sequence:

targetPose = zeroPose

movementData = {
    'id': 'movSeq1',
    'type': 'mov',
    'targetPose': targetPose,
    'maxVel': maxVel,
    'maxAccel': maxAccel
}
sim.callScriptFunction('remoteApi_movementDataFunction',scriptHandle,movementData)

# Execute first movement sequence:
sim.callScriptFunction('remoteApi_executeMovement',scriptHandle,'movSeq1')

# Wait until above movement sequence finished executing:
waitForMovementExecutedAsync('movSeq1')

# Send second and third movement sequence, where third one should execute
# immediately after the second one:
targetPose = yoke_rb

movementData = {
    'id': 'movSeq2',
    'type': 'mov',
    'targetPose': targetPose,
    'maxVel': maxVel,
    'maxAccel': maxAccel
}
sim.callScriptFunction('remoteApi_movementDataFunction',scriptHandle,movementData)
movementData = {
    'id': 'movSeq3',
    'type': 'mov',
    'targetPose': initialPose,
    'maxVel': maxVel,
    'maxAccel': maxAccel
}
sim.callScriptFunction('remoteApi_movementDataFunction',scriptHandle,movementData)

# Execute second and third movement sequence:
sim.callScriptFunction('remoteApi_executeMovement',scriptHandle,'movSeq2')
sim.callScriptFunction('remoteApi_executeMovement',scriptHandle,'movSeq3')

# Wait until above 2 movement sequences finished executing:
waitForMovementExecutedAsync('movSeq3')

sim.stopSimulation()

print('Program ended')
