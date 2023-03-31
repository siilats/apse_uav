from dataclasses import dataclass

from dacite import from_dict
import yaml
import json
import numpy as np
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import cv2


@dataclass
class CoppeliaConfig:
    yoke_board_y: float
    yoke_board_roll: float
    yoke_board_pitch: float
    yoke_board_yaw: float

    yoke_joint_0_x: float
    yoke_joint_0_y: float

    yoke_joint_0_pitch: float
    yoke_joint_0_yaw: float

    yoke_joint_1_x: float
    yoke_joint_1_y: float
    yoke_joint_1_z: float

    yoke_joint_1_pitch: float
    yoke_joint_1_yaw: float

    floor_level: float
    floor_height: float
    base_pitch: int
    base_yaw: float
    yoke_yaw: float
    cam_height: float
    gripper_board_x: float
    gripper_board_y: float
    gripper_board_z: float
    gripper_board_pitch: float
    gripper_board_yaw: float
    gripper_board_roll: float

@dataclass
class FramesConfig:
    step: int
    start: int
    end: int


@dataclass
class CapturingConfig:
    type: str
    source: str
    save_results: bool
    save_images: bool
    use_images: bool
    use_video: bool
    path_input_images: str
    path_input_video: str
    frames: FramesConfig
    coppelia: CoppeliaConfig
    marker_length: float
    use_boards: bool
    square_len: float
    marker_separation: float
    gamma: float
    grid_start: int
    grid_end: int
    base_car: int
    gripper: int
    moving_car: int
    coppelia_path: str

@dataclass
class DrawSettingsConfig:
    markers: bool
    marker_axes: bool
    id_pose_data: bool
    distance: bool
    leds: bool
    lines: bool
    points: bool


@dataclass
class SetupConfig:
    show_image: bool
    draw_settings: DrawSettingsConfig
    use_coppelia_sim: bool

@dataclass
class ModelConfig:
    capturing: CapturingConfig
    setup: SetupConfig

    @staticmethod
    def from_config(config):
        return from_dict(data_class=ModelConfig, data=config)

    @staticmethod
    def from_yaml_file(path):
        with open(path, "r") as f:
            config_yaml = yaml.load(f, Loader=yaml.UnsafeLoader)
        return from_dict(data_class=ModelConfig, data=config_yaml)


def transformYokeDistToAngle(yoke_dist):
    dist = abs(12.1 - yoke_dist)

    angle = dist / 2 * -15

    return angle / 360 * 2 * 3.1415

def readCameraParams(config):
    #read camera parameters from file
    with open(config.source, "r") as file:
        cam_params = json.load(file)

    #camera matrix
    mtx = np.array(cam_params["mtx"])

    #distortion coefficients
    dist = np.array(cam_params["dist"])

    return mtx, dist


def setArucoParameters():
    parameters = aruco.DetectorParameters()

    #set values for Aruco detection parameters
    parameters.minMarkerPerimeterRate = 0.01 #enables detection from higher altitude
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.33
    parameters.errorCorrectionRate = 2.0 #much more detections from high altitude, but FP happen sometimes
    parameters.aprilTagMinClusterPixels = 100 #less candidates to encode ID
    parameters.aprilTagMaxNmaxima = 5
    parameters.aprilTagCriticalRad = 20*np.pi/180 #much less candidates to encode ID
    parameters.aprilTagMaxLineFitMse = 1
    parameters.aprilTagMinWhiteBlackDiff = 100 #faster detection, but in bad contrast problems may happen
    #parameters.aprilTagQuadDecimate = 1.5 #huge detection time speedup, but at the cost of fewer detections and worse accuracy

    #default set of all Aruco detection parameters
    #parameters.adaptiveThreshWinSizeMin = 3
    #parameters.adaptiveThreshWinSizeMax = 23
    #parameters.adaptiveThreshWinSizeStep = 10
    #parameters.adaptiveThreshConstant = 7
    #parameters.minMarkerPerimeterRate = 0.03
    #parameters.maxMarkerPerimeterRate = 4
    #parameters.polygonalApproxAccuracyRate = 0.03
    #parameters.minCornerDistanceRate = 0.05
    #parameters.minDistanceToBorder = 3
    #parameters.minMarkerDistanceRate = 0.05
    #parameters.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    #parameters.cornerRefinementWinSize = 5
    #parameters.cornerRefinementMaxIterations = 30
    #parameters.cornerRefinementMinAccuracy = 0.1
    #parameters.markerBorderBits = 1
    #parameters.perspectiveRemovePixelPerCell = 4
    #parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    #parameters.maxErroneousBitsInBorderRate = 0.35
    #parameters.minOtsuStdDev = 5.0
    #parameters.errorCorrectionRate = 0.6
    #parameters.aprilTagMinClusterPixels = 5
    #parameters.aprilTagMaxNmaxima = 10
    #parameters.aprilTagCriticalRad = 10*np.pi/180
    #parameters.aprilTagMaxLineFitMse = 10
    #parameters.aprilTagMinWhiteBlackDiff = 5
    #parameters.aprilTagDeglitch = 0
    #parameters.aprilTagQuadDecimate = 0
    #parameters.aprilTagQuadSigma = 0
    #parameters.detectInvertedMarker = False

    return parameters


def setAverageMarkerSize(N_avg):
    #temp variables for averaging marker size
    msp1_avg = np.zeros((N_avg,1))
    msp2_avg = np.zeros((N_avg,1))
    msp3_avg = np.zeros((N_avg,1))
    msp4_avg = np.zeros((N_avg,1))

    return msp1_avg, msp2_avg, msp3_avg, msp4_avg

def preprocessFrame(frame, mapx, mapy, lookUpTable):
    #remove distortion from camera
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    #perform gamma correction
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lab[...,0] = cv2.LUT(lab[...,0], lookUpTable)
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return frame

def detectArucoMarkers(gray, parameters):
    #use predefined Aruco markers dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    #detect markers with APRILTAG method
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    detector = aruco.ArucoDetector(aruco_dict)
    detector.setDetectorParameters(parameters)

    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    return corners, ids

def getMarkerData(corners, rvec, cx_prev, cy_prev, markerLength):
    #marker centre x and y
    cx = int(corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4
    cy = int(corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4

    #marker size in pixels, cosine of yaw angle, sine of yaw angle
    msp = ((np.sqrt(np.power((corners[1][0]-corners[0][0]),2) + np.power((corners[1][1]-corners[0][1]),2)) +
            np.sqrt(np.power((corners[2][0]-corners[1][0]),2) + np.power((corners[2][1]-corners[1][1]),2)) +
            np.sqrt(np.power((corners[3][0]-corners[2][0]),2) + np.power((corners[3][1]-corners[2][1]),2)) +
            np.sqrt(np.power((corners[0][0]-corners[3][0]),2) + np.power((corners[0][1]-corners[3][1]),2))) / 4)

    #distance in metres between marker of the same ID on subsequent frames
    if cx_prev is not None and cy_prev is not None:
        diff = np.sqrt(np.power(cx_prev-cx,2) + np.power(cy_prev-cy,2)) * markerLength / msp
    else:
        diff = 0
    r = R.from_rotvec(rvec)
    ang = r.as_euler('zxy', degrees=True)[0]
    return abs(cx), abs(cy), msp, diff, ang

def calculateAverageMarkerSize(msp_avg, msp, N_avg):
    #write last measured marker size into table
    if(N_avg == 1):
        msp_avg = msp
    elif(N_avg > 1 and isinstance(N_avg, int)):
        for j in range(N_avg-1):
            msp_avg[j] = msp_avg[j+1]
        msp_avg[N_avg-1] = msp

    #calculate the average and rescale marker size
    nonzero = np.count_nonzero(msp_avg)
    size_corr = np.sum(msp_avg)/(msp*nonzero)
    msp = msp * size_corr

    return size_corr, msp

def markerLengthCorrection(altitude, markerLengthOrg, marker_div, div):
    #use correction of marker size based on current altitude
    return markerLengthOrg * (1 - 0.00057 * altitude/marker_div) / div

def printDataOnImage(corners, tvec, rvec, ids, marker_div, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    r = R.from_rotvec(rvec)

    #calculate real altitude to be printed
    tvec_temp = tvec.copy()
    tvec_temp[2] = tvec_temp[2]/marker_div

    #calculate angles and position and convert them to text
    ang = 'R = ' + str([round(r.as_euler('zxy', degrees=True)[0],2),
                        round(r.as_euler('zxy', degrees=True)[1],2),
                        round(r.as_euler('zxy', degrees=True)[2],2)]) + 'deg'
    pos = 't = ' + str([round(j,3) for j in tvec_temp]) + 'm'
    id = 'ID = ' + str(ids)

    #calculate the position where the text will be placed on image
    position = tuple([int(corners[0]-150), int(corners[1]+150)])
    position_ang = tuple([int(position[0]-0), int(position[1]+50)])
    position_id = tuple([int(position[0]-0), int(position[1]-50)])

    #write the text onto the image
    cv2.putText(frame, id, position_id, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, pos, position, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, ang, position_ang, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

def centroidFromAruco(veh_coords, tvec, rvec, size_corr, mtx, dist):
    #project measured centroid of the vehicle wrt. Aruco marker onto image
    imgpts, _ = cv2.projectPoints(veh_coords, rvec, tvec/size_corr, mtx, dist)
    imgpts = np.maximum(0,np.int32(imgpts).reshape(-1,2))

    return imgpts

def calculateDistance(lidar, aruco, markerLength, msp1, msp4):
    #calculate distances to Aruco marker and bbox of the vehicle
    d_aruco = np.sqrt((lidar[0][0]-aruco[0][0]) * (lidar[0][0]-aruco[0][0]) + (lidar[0][1]-aruco[0][1]) * (lidar[0][1]-aruco[0][1]))

    #convert distances from pixels to metres
    dist_aruco = d_aruco * markerLength / ((msp4+msp1)/2)

    return dist_aruco

def drawLinesOnImage(source, cx, cy, dist_aruco, frame, draw_settings, ang1=0, ang4=0):

    #draw the line from source of the measurement to the centre of vehicle' Aruco marker
    cv2.line(frame, (int(source[0][0]), int(source[0][1])), (int(cx), int(cy)), (0,0,255), 5)

    if draw_settings.distance:
        font = cv2.FONT_HERSHEY_SIMPLEX

        #calculate angles and position and convert them to text
        dist_aruco = str(round(dist_aruco,1)) + ','
        angle = str(round(ang1 - ang4, 1)) + ' degrees'

        #calculate the position where the text will be placed on image
        position_red = tuple([int((source[0][0]+cx)/2-200), int((source[0][1]+cy)/2)-50])
        position_yellow = tuple([int((source[0][0]+cx)/2+50), int((source[0][1]+cy)/2)-50])

        #write the text onto the image
        cv2.putText(frame, dist_aruco, position_red, font, 3.0, (0, 0, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, angle, position_yellow, font, 3.0, (0, 255, 255), 6, cv2.LINE_AA)


def gamma_lookup(config):
    lookUpTable = np.empty((1, 256), np.uint8)  # look-up table for config.gamma correction
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, config.gamma) * 255.0, 0, 255)

    return lookUpTable

def convert_angles(rvec):
    r = R.from_rotvec(rvec)
    ang = r.as_euler('zxy', degrees=True)
    return ang


def pick_rvec(rvecs, tvecs):
    generic_ang1 = convert_angles(rvecs[0].ravel())
    generic_ang2 = convert_angles(rvecs[1].ravel())
    if abs(generic_ang1[2] - 180) < abs(generic_ang2[2] - 180):
        rvectmp = rvecs[0].ravel()
        tvectmp = tvecs[0].ravel()
    else:
        rvectmp = rvecs[1].ravel()
        tvectmp = tvecs[1].ravel()

    return rvectmp, tvectmp

def pick_rvec2(rvecs, tvecs):
    generic_ang1 = convert_angles(rvecs[0].ravel())
    generic_ang2 = convert_angles(rvecs[1].ravel())
    if abs(generic_ang1[2] - 180) < abs(generic_ang2[2] - 180):
        rvectmp = rvecs[0]
        tvectmp = tvecs[0]
    else:
        rvectmp = rvecs[1]
        tvectmp = tvecs[1]

    return rvectmp, tvectmp

def pick_rvec_wrong(rvecs, tvecs):
    generic_ang1 = convert_angles(rvecs[0].ravel())
    generic_ang2 = convert_angles(rvecs[1].ravel())
    if abs(generic_ang1[2] - 180) > abs(generic_ang2[2] - 180):
        rvectmp = rvecs[0]
        tvectmp = tvecs[0]
    else:
        rvectmp = rvecs[1]
        tvectmp = tvecs[1]

    return rvectmp, tvectmp

def detect_charuco_board(config, gray, aruco_dict, parameters):
    base_board_size = (3, 3)
    marker_len = config.marker_length
    base_board = aruco.CharucoBoard(base_board_size, squareLength=config.square_len,
                                    markerLength=config.marker_length, dictionary=aruco_dict,
                                    ids=np.arange(4))

    base_detector = aruco.CharucoDetector(base_board)
    temp = base_detector.getCharucoParameters()
    temp.minMarkers = 0
    temp.tryRefineMarkers = True
    base_detector.setDetectorParameters(parameters)
    base_detector.setCharucoParameters(temp)
    # define the planar aruco board and its detector
    # the default ids is np.arange(24, 27)

    #corners1, ids1, rejectedImgPoints1 = cv2.aruco.detectMarkers(gray, aruco_dict)
    #detector = aruco.ArucoDetector(aruco_dict)
    #detector.setDetectorParameters(parameters)
    #corners, ids, rejected_img_points = detector.detectMarkers(gray)
    #diamondCorners1, diamondIds1, corners, ids = base_detector.detectDiamonds(gray)
    # num_corners21, corners21, ids21 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, base_board)


    base_corners, base_ids, corners, ids = base_detector.detectBoard(gray)

    return base_corners, base_ids, corners, ids, base_board


def create_grid_board(config, aruco_dict, gray, corners, ids, mtx, dist, start_id, end_id):
    yoke_board_size = (end_id - start_id + 1, 1)
    yoke_board = aruco.GridBoard(yoke_board_size, markerLength=config.marker_length,
                                 markerSeparation=config.marker_separation, dictionary=aruco_dict,
                                 ids=np.arange(start_id, end_id+1))
    yoke_detector = aruco.ArucoDetector(dictionary=aruco_dict,
                                        refineParams=aruco.RefineParameters())
    # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)

    ids_for_planar_board = np.argwhere((ids.ravel() >= start_id) & (ids.ravel() <= end_id)).ravel()
    yoke_board_corners, yoke_board_ids, yoke_rejectedCorners, yoke_recoveredIdxs = \
        yoke_detector.refineDetectedMarkers(gray, yoke_board,
                                            np.asarray(corners)[ids_for_planar_board],
                                            ids[ids_for_planar_board].ravel(),
                                            cameraMatrix=mtx,
                                            rejectedCorners=None,
                                            distCoeffs=dist)
    yoke_obj_points, yoke_img_points = yoke_board.matchImagePoints(yoke_board_corners, yoke_board_ids)

    return yoke_board_corners, yoke_obj_points, yoke_img_points, yoke_board

def matchImagePointsforcharuco(charuco_corners, charuco_ids, base_board):
    base_obj_pts = []
    base_img_pts = []
    for i in range(0, len(charuco_ids)):
        index = charuco_ids[i]
        base_obj_pts.append(base_board.getChessboardCorners()[index])
        base_img_pts.append(charuco_corners[i])

    base_obj_pts = np.array(base_obj_pts)
    base_img_pts = np.array(base_img_pts)

    return base_obj_pts, base_img_pts

def find_poses(ids, corners, mtx, dist, obj_points2):
    tvec = np.zeros((len(ids), 3))
    rvec = np.zeros((len(ids), 3))
    for i in range(len(ids)):
        # only markers with ID={1,2,3,4} are used at this moment
        # rvectmp=rvec[i][0] #compartible w previous version
        # tvectmp=tvec[i][0] #compartible w previous version
        flag, rvecs, tvecs, r2 = cv2.solvePnPGeneric(
            obj_points2, corners[i], mtx, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rvectmp, tvectmp = pick_rvec(rvecs, tvecs)
        tvec[i] = tvectmp
        rvec[i] = rvectmp
    return tvec, rvec

def draw_everything(draw_settings,corners,frame, mtx, dist, rvec, tvec, markerLength,id,marker_div=1.2):
    if draw_settings.markers:
        cv2.drawContours(frame, [np.maximum(0, np.int32(corners))], -1, (0, 255, 0), 3)
    if draw_settings.marker_axes:
        aruco.drawAxis(frame, mtx, dist, rvec, tvec, markerLength)
    if draw_settings.id_pose_data:
        printDataOnImage(corners[0], rvec, tvec, id, marker_div,frame)

def calculate_everything(config, moving_car_corners, tvec, rvec,
    cx4_prev, cy4_prev, k, msp4_avg, veh4_coords, veh4_dim, N_avg, mtx, dist ):
    cx4, cy4, msp, diff4, ang4 = getMarkerData(moving_car_corners, rvec,
                                               None if k == config.frames.start else cx4_prev,
                                               None if k == config.frames.start else cy4_prev, config.marker_length)  # get detected marker parameters
    size_corr4, msp4 = calculateAverageMarkerSize(msp4_avg, msp, N_avg)  # marker size averaging
    imgpts_veh4 = centroidFromAruco(veh4_coords, tvec, rvec,
                                    size_corr4, mtx, dist)  # calculate centroid of the vehicle wrt. Aruco marker
    return cx4, cy4, msp, diff4, ang4, size_corr4, msp4, imgpts_veh4
    # draw bounding box of the vehicle

def initial_coppelia(sim, baseBoard, yokeBoard, visionSensor, cc, gripperBoard, tip, yoke_joint0, yoke_joint1):
    floor_level = cc.floor_level
    x = 360 / (2 * np.pi)

    if cc.base_pitch == -90:
        sim.setObjectPosition(baseBoard, -1, [0, -floor_level, cc.floor_height])
    else:
        sim.setObjectPosition(baseBoard, -1, [0, -1.0000e-03, floor_level])
    sim.setObjectOrientation(baseBoard, -1, [cc.base_pitch / x, 0, cc.base_yaw / x])

    yoke_bg = sim.getObject('/yoke_background')
    sim.setObjectPosition(yokeBoard, yoke_bg, [0, cc.yoke_board_y, floor_level * 3])
    sim.setObjectOrientation(yokeBoard, yoke_bg, [cc.yoke_board_roll / x, cc.yoke_board_pitch, cc.yoke_board_yaw / x])

    sim.setJointPosition(yoke_joint0, 0)
    sim.setJointPosition(yoke_joint1, 15 / x)

    sim.setObjectPosition(gripperBoard, tip, [cc.gripper_board_x, cc.gripper_board_y, cc.gripper_board_z])
    sim.setObjectOrientation(gripperBoard, tip, [cc.gripper_board_pitch / x, cc.gripper_board_roll / x, cc.gripper_board_yaw / x])

    above_orientation = [-180 / x, 0, 180 / x]
    forward_orientation = [-1.1000e+02 / x, +2.0000e+01 / x, -1.8000e+02 / x]
    forward_position = [-5.0000e-01, -1.2000e+00, +1.0000e+00]
    # sim.yawPitchRollToAlphaBetaGamma(visionSensorHandle, 180.0, 0.0, -180.0)
    # alpha, beta, gamma = sim.alphaBetaGammaToYawPitchRoll(-180/360*2*3.1415, 0, -180/360*2*3.1415)
    sim.setObjectOrientation(visionSensor, -1, forward_orientation)
    sim.setObjectPosition(visionSensor, -1, forward_position)

def obj_points_square(markerLength):
    obj_points = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    obj_points2 = np.array([[-markerLength / 2, markerLength / 2, 0],
                            [markerLength / 2, markerLength / 2, 0],
                            [markerLength / 2, -markerLength / 2, 0],
                            [-markerLength / 2, -markerLength / 2, 0]])

    return obj_points, obj_points2


def invert_vec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.transpose(R)
    invTvec = -R @ tvec
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

def relative_position(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape(
        (3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = invert_vec(rvec2, tvec2)

    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(invRvec)
    R3 = R2 @ R1
    rvec3, _ = cv2.Rodrigues(R3)
    tvec3 = R2 @ tvec1 + invTvec

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec