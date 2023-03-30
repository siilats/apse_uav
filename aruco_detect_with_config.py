import cv2
from cv2 import aruco
import numpy as np
import json
import csv
import os

from func_timeout import func_timeout
from scipy.spatial.transform import Rotation as R
import time
from zmqRemoteApi import RemoteAPIClient
import argparse

from unitree import *

parser = argparse.ArgumentParser(description='Name of the config file')
parser.add_argument('--config', type=str,
                    help='config file', default="two_cars.yaml")

args = parser.parse_args()

model = ModelConfig.from_yaml_file(args.config)
setup = model.setup
coppelia_config = model.capturing.coppelia
draw_settings = setup.draw_settings
config = model.capturing

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
markerLength = config.marker_length #real size of the marker in metres

obj_points, obj_points2 = obj_points_square(markerLength)

parameters = setArucoParameters() #create Aruco detection parameters
mtx, dist = readCameraParams(config) #read camera parameters

[cx1_prev, cy1_prev, cx2_prev, cy2_prev, cx3_prev, cy3_prev, cx4_prev, cy4_prev, cx5_prev, cy5_prev] = np.zeros(10, dtype='int') #initialization of ArUco marker centres

if config.use_images:
    k = config.frames.start
    config.frames.end = len(os.listdir(config.path_input_images)) if config.frames.end is None else config.frames.end
    frame = cv2.imread(config.path_input_images + "/image_%04d.png" % config.frames.start)
elif config.use_video:
    video = cv2.VideoCapture(config.path_input_video)
    k = config.frames.start
    if config.frames.start >= 1 and video.isOpened():
        for i in range(config.frames.start):
            ret, frame = video.read()
            if ret == False:
                break
    config.frames.end = np.inf if config.frames.end is None else config.frames.end
    
height, width, channels = frame.shape
fov = 60.0
focal_length = width / (2 * np.tan(fov * np.pi / 360))
mtx = np.array([[focal_length, 0, width / 2],
                [0, focal_length, height / 2],
                [0, 0, 1]])
dist = None

if setup.use_coppelia_sim:
    client = RemoteAPIClient()
    try:
        sim = func_timeout(3, lambda: client.getObject('sim'))
    except:
        raise IOError("Failed to connect to Coppeliasim, either open it or set setup.use_coppelia_sim to false")
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    sim.loadScene(os.getcwd() + config.coppelia_path)
    visionSensor = sim.getObject('/Vision_sensor')

    baseBoard = sim.getObject('/base_board')
    yokeBoard = sim.getObject('/yoke_board')
    gripperBoard = sim.getObject('/gripper_board')
    tip = sim.getObject('/tip')
    yoke_joint0 = sim.getObject('/yoke_joint0')
    yoke_joint1 = sim.getObject('/yoke_joint1')

    #read 6 joints of the robot
    joints = []
    for i in range(6):
        joints.append(sim.getObject('/joint%d' % (i+1)))
        if i+1 == 2:
            sim.setJointPosition(joints[i], 90/180*np.pi)
        elif i+1 == 3:
            sim.setJointPosition(joints[i], -90/180*np.pi)
        else:
            sim.setJointPosition(joints[i], 0)

    initial_coppelia(sim, baseBoard, yokeBoard, visionSensor, coppelia_config, gripperBoard, tip, yoke_joint0, yoke_joint1)

    defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
    sim.setInt32Param(sim.intparam_idle_fps, 0)

    # sim.handleVisionSensor(visionSensorHandle)

    # Run a simulation in stepping mode:
    client.setStepping(True)
    sim.startSimulation()
    client.step()
    sim.addLog(sim.verbosity_scriptinfos, "all set up ---------------------------")

camera_location = None
camera_orientation = None
base_car_detected = 0
moving_car_detected = 0
gripper_detected = 0

#iterate over frames
while k <= config.frames.end and (config.use_images or (config.use_video and video.isOpened())):
    #read frame from image or video
    if config.use_images:
        frame = cv2.imread(config.path_input_images + "/image_%04d.png" % k)
    elif config.use_video:
        ret, frame = video.read()
        if ret == False:
            break
    #convert image to grayscale and detect Aruco markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if config.use_boards:
        base_corners, base_ids, corners, ids, base_board = detect_charuco_board(config, gray, aruco_dict, parameters)
    else:
        corners, ids = detectArucoMarkers(gray, parameters)

    if (ids is not None) and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    # write me adaptive grayscale in opencv
    idx = np.argsort(ids.ravel())
    corner_not_tuple = np.array(corners)[idx]
    corners = tuple(corner_not_tuple)
    ids = ids[idx]

    tvec = np.zeros((len(ids),3))
    rvec = np.zeros((len(ids),3))
    for i in range(len(ids)):
        flag, rvecs, tvecs, r2 = cv2.solvePnPGeneric(
            obj_points2, corners[i], mtx, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rvectmp, tvectmp = pick_rvec(rvecs, tvecs)
        tvec[i] = tvectmp
        rvec[i] = rvectmp

    #if any marker was detected
    if np.all(ids != None):
        if config.use_boards:
            yoke_board_corners, yoke_obj_points, yoke_img_points, yoke_board = \
                create_grid_board(config, aruco_dict, gray, corners, ids, mtx, dist, config.grid_start, config.grid_end)
            yoke_flag, yoke_rvecs, yoke_tvecs, yoke_r2 = cv2.solvePnPGeneric(
                yoke_obj_points, yoke_img_points, mtx, dist,
                flags=cv2.SOLVEPNP_IPPE)
            rvectmp, tvectmp = pick_rvec(yoke_rvecs, yoke_tvecs)
            tvec[0] = tvectmp
            rvec[0] = rvectmp

            base_obj_points, base_img_points = matchImagePointsforcharuco(base_corners, base_ids, base_board)

            base_flag, base_rvecs, base_tvecs, base_reproj_error = \
                cv2.solvePnPGeneric(base_obj_points, base_img_points, mtx, dist, flags=cv2.SOLVEPNP_IPPE)


            rvectmp, tvectmp = pick_rvec(base_rvecs, base_tvecs)
            tvec[1] = tvectmp
            rvec[1] = rvectmp

            cv2.drawFrameAxes(frame, mtx, dist, rvec[1], tvec[1], 1)

            if setup.use_coppelia_sim:
                R, _ = cv2.Rodrigues(rvec[1])

                # Invert the rotation matrix and translation vector
                R_inv = np.transpose(R)  # Transpose of R is the inverse for orthogonal matrices
                tvec_inv = -np.dot(R_inv, tvec[1])

                # Convert the inverse rotation matrix to an inverse rotation vector
                rvec_inv, _ = cv2.Rodrigues(R_inv)

                camera_orientation = rvec_inv
                camera_location = tvec_inv


                # First time detecting the base board move the camera and leave base board at 0,0,0
                if base_car_detected == 0:
                    r1 = R.from_rotvec(camera_orientation)
                    baseBoard_orientation = r1.as_euler('zxy', degrees=True)[0]
                    camera_location_coppelia = [-camera_location[0], 0, camera_location[2]]
                    sim.setObjectPosition(visionSensor, -1, camera_location_coppelia)

            gripper_board_corners, gripper_obj_points, gripper_img_points, gripper_board = create_grid_board(config, aruco_dict,
                                                            gray, corners, ids,mtx, dist, config.gripper, config.gripper+1)
            gripper_flag, gripper_rvecs, gripper_tvecs, ggripper_r2 = cv2.solvePnPGeneric(
                gripper_obj_points, gripper_img_points, mtx, dist,
                flags=cv2.SOLVEPNP_IPPE)
            rvectmp, tvectmp = pick_rvec(gripper_rvecs, gripper_tvecs)
            tvec[2] = tvectmp
            rvec[2] = rvectmp


            ids[0][0] = config.moving_car
            ids[1][0] = config.base_car
            ids[2][0] = config.gripper

            moving_car_detected = 1
            base_car_detected = 1
            gripper_detected = 1


            cx1, cy1, msp1, diff1, ang1 = getMarkerData(base_corners.squeeze(), rvec[0],
                                                             None if k == config.frames.start else cx1_prev,
                                                             None if k == config.frames.start else cy1_prev, markerLength)  # get detected marker parameters

            cx4, cy4, msp4, diff4, ang4 = getMarkerData(yoke_board_corners[0].squeeze(), rvec[1],
                                                             None if k == config.frames.start else cx4_prev,
                                                             None if k == config.frames.start else cy4_prev, markerLength)  # get detected marker parameters

            cx5, cy5, msp5, diff5, ang5 = getMarkerData(gripper_board_corners[0].squeeze(), rvec[2],
                                                        None if k == config.frames.start else cx5_prev,
                                                        None if k == config.frames.start else cy5_prev,
                                                        markerLength)  # get detected marker parameters

        else:
            # find the index of the moving car from ids using argwhere
            base_car_idx = (ids.ravel() == config.base_car)
            moving_car_idx = (ids.ravel() == config.moving_car)

            if np.any(base_car_idx):
                base_car_index = np.argwhere(base_car_idx).ravel()[0]
                base_car_corners = corners[base_car_index][0]
                if setup.use_coppelia_sim:
                    camera_orientation = rvec[base_car_index]
                    camera_location = tvec[base_car_index]

                    # First time detecting the base board move the camera and leave base board at 0,0,0
                    if base_car_detected == 0:
                        r1 = R.from_rotvec(camera_orientation)
                        baseBoard_orientation = r1.as_euler('zxy', degrees=True)[0]
                        camera_location_coppelia = [-camera_location[0], 0, camera_location[2]]
                        sim.setObjectPosition(visionSensor, -1, camera_location_coppelia)
                base_car_detected = 1
                cx1, cy1, msp1, diff1, ang1 = getMarkerData(base_car_corners, rvec[base_car_index],
                                                           None if k == config.frames.start else cx1_prev,
                                                           None if k == config.frames.start else cy1_prev,
                                                           config.marker_length)  # get detected marker parameters
                draw_everything(draw_settings, base_car_corners, frame, mtx, dist, rvec[base_car_index],
                                tvec[base_car_index], markerLength, config.base_car)
                cx1_prev, cy1_prev = cx1, cy1  # save position of the marker in the image
            if np.any(moving_car_idx):
                moving_car_index = np.argwhere(moving_car_idx).ravel()[0]
                moving_car_corners = corners[moving_car_index][0]
                moving_car_detected = 1
                if setup.use_coppelia_sim and camera_location is not None:
                    # move the yoke marker
                    tvectmp_cp = tvec[moving_car_index] - camera_location
                    tvectmp_cp[2] = coppelia_config.floor_level

                    sim.setObjectPosition(yokeBoard, -1, [tvectmp_cp[0], -tvectmp_cp[1], tvectmp_cp[2]])

                    r4 = R.from_rotvec(rvec[moving_car_index])
                    moving_angle = r4.as_euler('zxy', degrees=True)[0]
                    r1 = R.from_rotvec(camera_orientation)
                    baseBoard_orientation = r1.as_euler('zxy', degrees=True)[0]
                    correction = -90 - baseBoard_orientation  # -7
                    final_angle = 180 - moving_angle + correction

                    # we have -90 and then angle4 is 120 and we want answe 45
                    sim.setObjectOrientation(yokeBoard, -1,
                                                  [180 / 360 * 2 * 3.1415, 0, final_angle / 360 * 2 * 3.1415])

                    img, resX, resY = sim.getVisionSensorCharImage(visionSensor)
                    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
                    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
                    img_name = "image_{}.png".format(k)
                    corners_t, ids_t = detectArucoMarkers(img, parameters)
                    cv2.imwrite("test_video/" + img_name, img)
                
                cx4, cy4, msp4, diff4, ang4 = getMarkerData(moving_car_corners, rvec[moving_car_index],
                                                           None if k == config.frames.start else cx1_prev,
                                                           None if k == config.frames.start else cy1_prev,
                                                           config.marker_length)  # get detected marker parameters
                draw_everything(draw_settings, moving_car_corners, frame, mtx, dist, rvec[moving_car_index],
                                tvec[moving_car_index], markerLength, config.moving_car)
                cx4_prev, cy4_prev = cx4, cy4 #save position of the marker in the image

        if moving_car_detected and base_car_detected:
            dist_veh1_aruco = calculateDistance(np.float32([[cx4, cy4]]), np.float32([[cx1, cy1]]), markerLength, msp1, msp4)  # calculate distances in metres for Aruco method
            if draw_settings.lines:
                drawLinesOnImage(np.float32([[cx4, cy4]]), cx1, cy1, dist_veh1_aruco, frame, draw_settings, ang1, ang4)
                # drawLinesOnImage(np.float32([[cx5, cy5]]), cx1, cy1, dist_veh1_aruco, frame, draw_settings, ang1,ang5)

        if setup.use_coppelia_sim:
            client.step()
    #show results on image
    if setup.show_image:
        cv2.namedWindow("Detection result", cv2.WINDOW_NORMAL)
        cv2.imshow("Detection result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #increment frame number
    k = k + config.frames.step

    #skip frames from video
    if config.use_video:
        for i in range(config.frames.step-1):
            ret, frame = video.read()
            if ret == False:
                break

if config.use_video:
    video.release()

if setup.show_image:
    cv2.destroyAllWindows()
