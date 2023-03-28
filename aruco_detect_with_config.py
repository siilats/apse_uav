import cv2
from cv2 import aruco
import numpy as np
import json
import csv
import os
from scipy.spatial.transform import Rotation as R


from unitree import *

model = ModelConfig.from_yaml_file('original_cars.yaml')
setup = model.setup
draw_settings = setup.draw_settings
config = model.capturing

#number of frames to be used for marker size averaging, recommended is 1
N_avg = 1
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

markerLengthOrg = config.marker_length #real size of the marker in metres, this value does not change in algorithm
markerLength = markerLengthOrg #real size of the marker in metres, this value changes in algorithm
 #correction for altitude estimation from marker
div = 1.013 #additional correction for distance calculation (based on altitude test)
DIFF_MAX = 2/3 * config.frames.step * 2 #maximum displacement of ArUco centre between frames with vehicle speed of 72 km/h = 20 m/s

obj_points = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
obj_points2 = np.array([[-markerLength / 2 , markerLength /  2, 0],
                        [markerLength / 2, markerLength / 2, 0],
                        [markerLength / 2, -markerLength / 2, 0],
                        [-markerLength / 2, -markerLength / 2, 0]])

parameters = setArucoParameters() #create Aruco detection parameters
mtx, dist = readCameraParams(config) #read camera parameters
msp1_avg, msp2_avg, msp3_avg, msp4_avg = setAverageMarkerSize(N_avg) #initialization of marker size averaging variables
detected_ID_prev = [0,0,0,0] #initialization of vehicle detection state on previous frame
[cx1_prev, cy1_prev, cx2_prev, cy2_prev, cx3_prev, cy3_prev, cx4_prev, cy4_prev] = np.zeros(8, dtype='int') #initialization of ArUco marker centres

lookUpTable = gamma_lookup(config)

#vehicle's centroid wrt. Aruco marker in metres
veh4_coords = np.float32([[0,0.07,0]])
veh1_coords = np.float32([[0,0.42,0]])
veh2_coords = np.float32([[0,0.59,0]])
veh3_coords = np.float32([[0,0.58,0]])

#initialize values if images are used
if config.use_images:
    k = config.frames.start
    config.frames.end = len(os.listdir(config.path_input_images)) if config.frames.end is None else config.frames.end
    frame = cv2.imread(config.path_input_images + "/image_%04d.png" % config.frames.start)

#initialize values if video is used
elif config.use_video:
    video = cv2.VideoCapture(config.path_input_video)
    k = config.frames.start
    if config.frames.start > 1 and video.isOpened():
        for i in range(config.frames.start-1):
            ret, frame = video.read()
            if ret == False:
                break
    config.frames.end = np.inf if config.frames.end is None else config.frames.end
    
height, width, channels = frame.shape

#calculate maps for undistortion
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (width, height), 5)

#real vehicle dimensions in metres wrt. Aruco marker: back, front, left, right
veh4_dim = [-2.35, 2.49, -0.86, 0.86]
veh1_dim = [-1.95, 2.8, -0.9, 0.9]
veh2_dim = [-1.68, 2.86, -0.87, 0.87]
veh3_dim = [-1.32, 2.48, -0.86, 0.86]



#iterate over frames
while k <= config.frames.end and (config.use_images or (config.use_video and video.isOpened())):
    #read frame from image or video
    if config.use_images:
        frame = cv2.imread(config.path_input_images + "/image_%04d.png" % k)
    elif config.use_video:
        ret, frame = video.read()
        if ret == False:
            break

    detected_ID = [0,0,0,0] #by default no vehicle is detected in image

    #frame preprocessing - camera distortion removal and config.gamma correction
    frame = preprocessFrame(frame, mapx, mapy, lookUpTable)

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
    corners = tuple(np.array(corners)[idx])
    ids = ids[idx]

    # write me adaptive grayscale in opencv


    #%%====================================
    #MARKER DETECTION AND POINTS CALCULATIONS
    tvec = np.zeros((100,3))
    rvec = np.zeros((100,3))
    #if any marker was detected
    if np.all(ids != None):
        if config.use_boards:
            yoke_board_corners, yoke_obj_points, yoke_img_points, yoke_board = create_grid_board(config, aruco_dict, gray, corners, ids, mtx, dist)
            yoke_flag, yoke_rvecs, yoke_tvecs, yoke_r2 = cv2.solvePnPGeneric(
                yoke_obj_points, yoke_img_points, mtx, dist,
                flags=cv2.SOLVEPNP_IPPE)
            rvectmp, tvectmp = pick_rvec(yoke_rvecs, yoke_tvecs)
            tvec[0] = tvectmp
            rvec[0] = rvectmp

            base_obj_points, base_img_points = matchImagePointsforcharuco(base_corners, base_ids, base_board)

            # bug in cv.solvePnPGeneric
            base_flag, base_rvecs, base_tvecs, base_reproj_error = cv2.solvePnPGeneric(base_obj_points,
                                                                                       base_img_points,
                                                                                       mtx,
                                                                                       dist,
                                                                                       flags=cv2.SOLVEPNP_IPPE)

            rvectmp, tvectmp = pick_rvec(base_rvecs, base_tvecs)
            tvec[3] = tvectmp
            rvec[3] = rvectmp
            ids[3][0] = 4
            ids[2][0] = 0
            ids[1][0] = 0
            ids[0][0] = 1
            moving_car_detected = 1
            base_car_detected = 1

            cx1, cy1, msp1, diff1, ang1 = getMarkerData(base_corners.squeeze(), rvec[0],
                                                             None if k == config.frames.start else cx1_prev,
                                                             None if k == config.frames.start else cy1_prev)  # get detected marker parameters

            cx4, cy4, msp4, diff4, ang4 = getMarkerData(yoke_board_corners[0].squeeze(), rvec[3],
                                                             None if k == config.frames.start else cx4_prev,
                                                             None if k == config.frames.start else cy4_prev)  # get detected marker parameters

            size_corr1, msp1 = calculateAverageMarkerSize(msp1_avg, msp1)  # marker size averaging
            size_corr4, msp4 = calculateAverageMarkerSize(msp4_avg, msp4)  # marker size averaging

        else:
            # find the index of the moving car from ids using argwhere
            base_car_idx = (ids.ravel() == config.base_car)
            moving_car_idx = (ids.ravel() == config.moving_car)

            if np.any(base_car_idx):
                base_car_index = np.argwhere(base_car_idx).ravel()[0]
                base_car_corners = corners[base_car_index][0]
                base_car_detected = 1
                cx1, cy1, msp, diff1, ang1, size_corr1, msp1, imgpts_veh1 = \
                    calculate_everything(config, base_car_corners,
                                         tvec[base_car_index], rvec[base_car_index],
                                         cx1_prev, cy1_prev, k, msp1_avg, veh1_coords, veh1_dim, N_avg, mtx, dist)
                draw_everything(draw_settings, base_car_corners, frame, mtx, dist, rvec[base_car_index],
                                tvec[base_car_index], markerLength, config.base_car)
                cx1_prev, cy1_prev = cx1, cy1  # save position of the marker in the image

            if np.any(base_car_idx):
                moving_car_index = np.argwhere(moving_car_idx).ravel()[0]
                moving_car_corners = corners[moving_car_index][0]
                moving_car_detected = 1
                cx4, cy4, msp, diff4, ang4, size_corr4, msp4, imgpts_veh4 = \
                    calculate_everything(config, moving_car_corners,
                                         tvec[moving_car_index], rvec[moving_car_index],
                                         cx4_prev, cy4_prev, k, msp4_avg, veh4_coords, veh4_dim, N_avg, mtx, dist)
                draw_everything(draw_settings, moving_car_corners, frame, mtx, dist, rvec[moving_car_index],
                                tvec[moving_car_index], markerLength, config.moving_car)
                cx4_prev, cy4_prev = cx4, cy4 #save position of the marker in the image

            if moving_car_detected and base_car_detected:
                dist_veh1_aruco = calculateDistance(np.float32([[cx4, cy4]]),
                                                                          np.float32([[cx1, cy1]]),
                                                                          markerLength, msp4,
                                                                          msp1)  # calculate distances in metres for Aruco method
                if draw_settings.lines:
                    drawLinesOnImage(np.float32([[cx4, cy4]]), cx1, cy1, dist_veh1_aruco, frame, draw_settings, ang1, ang4)  #

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
