# Make sure to have CoppeliaSim running, with followig scene loaded:
#
# scenes/messaging/ikMovementViaRemoteApi.ttt
#
# Do not launch simulation, then run this script
import random

import numpy as np
from scipy.spatial.transform import Rotation as R

from zmqRemoteApi import RemoteAPIClient

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

tipHandle = sim.getObject('/robot_arm/tip')
targetHandle = sim.getObject('/robot_arm/target')

# Set-up some movement variables:
maxVel = 0.1
maxAccel = 0.01
maxJerk = 80


# Start simulation:
sim.startSimulation()

def cb(pose,vel,accel,handle):
    sim.setObjectPose(handle,-1,pose)


# Send movement sequences:
initialPose = sim.getObjectPose(tipHandle, -1)
ini_coor_x = initialPose[0]
ini_coor_y = initialPose[1]
ini_coor_z = initialPose[2]
ini_quat_x = initialPose[3]
ini_quat_y = initialPose[4]
ini_quat_z = initialPose[5]
ini_quat_w = initialPose[6]

ini_euler = R.from_quat([ini_quat_x, ini_quat_y, ini_quat_z, ini_quat_w]).as_euler('zyx')

tar_coor_x = ini_coor_x + random.random()
tar_coor_y = ini_coor_y + random.random()
tar_coor_z = ini_coor_z + random.random()
tar_quat = R.from_euler('zyx', [ini_euler[0] + random.random(),
                                ini_euler[1] + random.random(), ini_euler[2] + random.random()]).as_quat()

targetPose = [tar_coor_x, tar_coor_y, tar_coor_z, tar_quat[0], tar_quat[1], tar_quat[2], tar_quat[3]]
sim.moveToPose(-1, initialPose, [maxVel], [maxAccel], [maxJerk], targetPose, cb, targetHandle, [1, 1, 1, 0.1])

# targetPose = [
#     0, 0, 0.85,
#     -0.7071068883, -6.252754758e-08, -8.940695295e-08, -0.7071067691
# ]
# sim.moveToPose(-1,sim.getObjectPose(tipHandle,-1),[maxVel],[maxAccel],[maxJerk],targetPose,cb,targetHandle,[1,1,1,0.1])

sim.moveToPose(-1,sim.getObjectPose(tipHandle,-1),[maxVel],[maxAccel],[maxJerk],initialPose,cb,targetHandle,[1,1,1,0.1])

sim.stopSimulation()

print('Program ended')