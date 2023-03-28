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

tipHandle = sim.getObject('/tip')
targetHandle = sim.getObject('/target')
robotBaseHandle = sim.getObject('/base')
yokeBoardHandle = sim.getObject('/yokeBoard')

# Set-up some movement variables:
maxVel = 0.1
maxAccel = 0.01
maxJerk = 80


# Start simulation:
sim.startSimulation()

def cb(pose,vel,accel,handle):
    sim.setObjectPose(handle,-1,pose)


# Send movement sequences:
tip_rb = sim.getObjectPose(tipHandle, robotBaseHandle)
ini_coor_x = tip_rb[0]
ini_coor_y = tip_rb[1]
ini_coor_z = tip_rb[2]
ini_quat_x = tip_rb[3]
ini_quat_y = tip_rb[4]
ini_quat_z = tip_rb[5]
ini_quat_w = tip_rb[6]

yoke_rb = sim.getObjectPose(yokeBoardHandle, robotBaseHandle)
#yoke_rb_euler = R.from_quat([yoke_rb[3], yoke_rb[4], yoke_rb[5], yoke_rb[6]]).as_euler('zyx')

yoke_rb_matrix = sim.poseToMatrix(yoke_rb)
yoke_rb_euler = sim.getEulerAnglesFromMatrix(yoke_rb_matrix)

#ini_euler = R.from_quat([ini_quat_x, ini_quat_y, ini_quat_z, ini_quat_w]).as_euler('zyx')

tar_coor_x = yoke_rb[0]
tar_coor_y = yoke_rb[1]
tar_coor_z = yoke_rb[2]
tar_quat = R.from_euler('zyx', [yoke_rb_euler[0],
                                yoke_rb_euler[1], yoke_rb_euler[2]]).as_quat()

target_rb = [tar_coor_x, tar_coor_y, tar_coor_z, tar_quat[0], tar_quat[1], tar_quat[2], tar_quat[3]]
sim.moveToPose(-1, tip_rb, [maxVel], [maxAccel], [maxJerk], target_rb, cb, targetHandle, [1, 1, 1, 0.1])

# targetPose = [
#     0, 0, 0.85,
#     -0.7071068883, -6.252754758e-08, -8.940695295e-08, -0.7071067691
# ]
# sim.moveToPose(-1,sim.getObjectPose(tipHandle,-1),[maxVel],[maxAccel],[maxJerk],targetPose,cb,targetHandle,[1,1,1,0.1])

sim.moveToPose(-1, sim.getObjectPose(tipHandle,-1), [maxVel], [maxAccel], [maxJerk], tip_rb, cb, targetHandle, [1, 1, 1, 0.1])

sim.stopSimulation()

print('Program ended')