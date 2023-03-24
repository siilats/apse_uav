# Make sure to have CoppeliaSim running, with followig scene loaded:
#
# scenes/messaging/movementViaRemoteApi.ttt
#
# Do not launch simulation, then run this script
import time

import math

from zmqRemoteApi import RemoteAPIClient

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

sim.stopSimulation()
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

executedMovId = 'notReady'

targetArm = '/blueArm'
# targetArm = '/redArm'

stringSignalName = targetArm + '_executedMovId'
objHandle = sim.getObject(targetArm)
scriptHandle = sim.getScript(sim.scripttype_childscript,objHandle)


def waitForMovementExecuted(id_):
    global executedMovId, stringSignalName
    while executedMovId != id_:
        s = sim.getStringSignal(stringSignalName)
        executedMovId = s
        time.sleep(0.1)

def waitForMovementExecutedAsync(id_):
    global executedMovId
    while executedMovId != id_:
        s = sim.waitForSignal(id_)
        if s is True:
            executedMovId = id_

# Set-up some movement variables:
mVel = 360 * math.pi / 180
mAccel = 3*360 * math.pi / 180
mJerk = 9*360 * math.pi / 180
maxVel = [mVel, mVel, mVel, mVel, mVel, mVel]
maxAccel = [mAccel, mAccel, mAccel, mAccel, mAccel, mAccel]
maxJerk = [mJerk, mJerk, mJerk, mJerk, mJerk, mJerk]
targetVel = [0, 0, 0, 0, 0, 0]

# Start simulation:
sim.startSimulation()

# Wait until ready:
waitForMovementExecuted('ready')

# Send first movement sequence:
targetConfig = [
    90 * math.pi / 180, 90 * math.pi / 180, -90 * math.pi / 180,
    90 * math.pi / 180, 90 * math.pi / 180, 90 * math.pi / 180
]
movementData = {
    'id': 'movSeq1',
    'type': 'mov',
    'targetConfig': targetConfig,
    'targetVel': targetVel,
    'maxVel': maxVel,
    'maxAccel': maxAccel,
    'maxJerk': maxJerk,
}
sim.callScriptFunction('remoteApi_movementDataFunction',scriptHandle,movementData)

# Execute first movement sequence:
sim.callScriptFunction('remoteApi_executeMovement',scriptHandle,'movSeq1')

# Wait until above movement sequence finished executing:
waitForMovementExecutedAsync('movSeq1')

# Send second and third movement sequence, where third one should execute
# immediately after the second one:
targetConfig = [
    -90 * math.pi / 180, 45 * math.pi / 180, 90 * math.pi / 180,
    135 * math.pi / 180, 90 * math.pi / 180, 90 * math.pi / 180
]
targetVel = [-60 * math.pi / 180, -20 * math.pi / 180, 0, 0, 0, 0]
movementData = {
    'id': 'movSeq2',
    'type': 'mov',
    'targetConfig': targetConfig,
    'targetVel': targetVel,
    'maxVel': maxVel,
    'maxAccel': maxAccel,
    'maxJerk': maxJerk,
}
sim.callScriptFunction('remoteApi_movementDataFunction',scriptHandle,movementData)
targetConfig = [0, 0, 0, 0, 0, 0]
targetVel = [0, 0, 0, 0, 0, 0]
movementData = {
    'id': 'movSeq3',
    'type': 'mov',
    'targetConfig': targetConfig,
    'targetVel': targetVel,
    'maxVel': maxVel,
    'maxAccel': maxAccel,
    'maxJerk': maxJerk,
}
sim.callScriptFunction('remoteApi_movementDataFunction',scriptHandle,movementData)

# Execute second and third movement sequence:
sim.callScriptFunction('remoteApi_executeMovement',scriptHandle,'movSeq2')
sim.callScriptFunction('remoteApi_executeMovement',scriptHandle,'movSeq3')

# Wait until above 2 movement sequences finished executing:
waitForMovementExecutedAsync('movSeq3')
sim.stopSimulation()

print('Program ended')
