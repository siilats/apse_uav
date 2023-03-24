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
objHandle = sim.getObject(targetArm)
scriptHandle = sim.getScript(sim.scripttype_childscript,objHandle)


def waitForMovementExecuted(id_):
    global executedMovId, stringSignalName
    while executedMovId != id_:
        s = sim.getStringSignal(stringSignalName)
        executedMovId = s


# Set-up some movement variables:
maxVel = 0.1
maxAccel = 0.01

base = [0, -0.65, 0.25, 0, 0, 0, 0]
yoke = [0.275, -0.255, 0.6, 0, 0, 0, 0]

# Start simulation:
sim.startSimulation()


# Wait until ready:
waitForMovementExecuted('ready')

# Get initial pose:
initialPose, initialConfig = sim.callScriptFunction('remoteApi_getPoseAndConfig',scriptHandle)

# Send first movement sequence:

targetPose = yoke

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
waitForMovementExecuted('movSeq1')

# Send second and third movement sequence, where third one should execute
# immediately after the second one:
targetPose = [
    0.2, 0, 0.4,
    -0.7071068883, -6.252754758e-08, -8.940695295e-08, -0.7071067691
]
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
waitForMovementExecuted('movSeq3')

sim.stopSimulation()

print('Program ended')
