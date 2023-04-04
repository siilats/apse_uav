import argparse
import math
import os
import platform
import time

if platform.system() == "Linux":
    import unitree_arm_interface
from cv2 import aruco
from func_timeout import func_timeout

from unitree import *
from zmqRemoteApi import RemoteAPIClient

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
markerLength = config.marker_length  # real size of the marker in metres

parameters = setArucoParameters()  # create Aruco detection parameters
mtx, dist = readCameraParams(config)  # read camera parameters

[cx1_prev, cy1_prev, cx2_prev, cy2_prev, cx3_prev, cy3_prev, cx4_prev, cy4_prev, cx5_prev, cy5_prev] = np.zeros(10,
                                                                                                                dtype='int')  # initialization of ArUco marker centres

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

client = RemoteAPIClient()
try:
    sim = func_timeout(3, lambda: client.getObject('sim'))
except:
    raise IOError("Failed to connect to Coppeliasim, either open it or set setup.use_coppelia_sim to false")
if setup.reset_sim:
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    sim.loadScene(os.getcwd() + config.coppelia_path)

visionSensor, baseBoard, baseBoardCorner, yokeBoard, yokeBoardCorner, gripperBoard, \
    gripperBoardCorner, tip, yoke_joint0, yoke_joint1, yoke_handle, target_handle, \
    tip_world, yoke_world, base_world, joints, z1_robot = standard_coppelia_objects(sim)

if setup.reset_sim:
    initial_coppelia(sim, baseBoard, yokeBoard, visionSensor, coppelia_config, gripperBoard, tip, yoke_joint0, yoke_joint1)

# sim.handleVisionSensor(visionSensorHandle)

# Run a simulation in stepping mode:
# client.setStepping(True)
# sim.startSimulation()
# client.step()



# iterate over frames
while k <= config.frames.end and (config.use_images or (config.use_video and video.isOpened())):
    # read frame from image or video
    if config.use_images:
        frame = cv2.imread(config.path_input_images + "/image_%04d.png" % k)
    elif config.use_video:
        ret, frame = video.read()
        if ret == False:
            break
    # convert image to grayscale and detect Aruco markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    base_corners, base_ids, corners, ids, base_board = detect_charuco_board(config, gray, aruco_dict, parameters)

    if np.all(ids == None) or len(ids) == 0:
        continue

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    aruco.drawDetectedCornersCharuco(frame, base_corners, base_ids)

    corners, ids = sort_corners(corners, ids)

    base_obj_points, base_img_points = matchImagePointsforcharuco(base_corners, base_ids, base_board)
    base_flag, base_rvecs, base_tvecs, base_reproj_error = \
        cv2.solvePnPGeneric(base_obj_points, base_img_points, mtx, dist, flags=cv2.SOLVEPNP_IPPE)
    obj_points, img_points = matchImagePointsforcharuco(base_corners, base_ids, base_board)
    base_rvec, base_tvec = pick_rvec_board(base_rvecs, base_tvecs)
    cv2.drawFrameAxes(frame, mtx, dist, base_rvec, base_tvec, 1, thickness=5)

    base_rvec_inv, base_tvec_inv = invert_vec(base_rvec, base_tvec)

    camera_w = sim.getObjectPosition(visionSensor, -1)
    base_board_corner_w = sim.getObjectPosition(baseBoardCorner, -1)
    camera_bb = sim.getObjectPosition(visionSensor, baseBoardCorner)
    camera_bb_pose = sim.getObjectPose(visionSensor, baseBoardCorner)
    camera_bb_orient = sim.getObjectOrientation(visionSensor, baseBoardCorner)
    cam_orient_coppelia = [base_rvec_inv[0][0] , base_rvec_inv[1][0], np.pi + base_rvec_inv[2][0]]

    camera_location_coppelia = [base_tvec_inv[0][0], base_tvec_inv[1][0], base_tvec_inv[2][0]]
    sim.setObjectPosition(visionSensor, baseBoardCorner, camera_location_coppelia)
    sim.setObjectOrientation(visionSensor, baseBoardCorner, cam_orient_coppelia)


    yoke_board_corners, yoke_obj_points, yoke_img_points, yoke_board = \
        create_grid_board(config, aruco_dict, gray, corners, ids, mtx, dist, config.grid_start, config.grid_end)
    yoke_flag, yoke_rvecs, yoke_tvecs, yoke_r2 = cv2.solvePnPGeneric(
        yoke_obj_points, yoke_img_points, mtx, dist,
        flags=cv2.SOLVEPNP_IPPE)
    yoke_rvec, yoke_tvec = pick_rvec_board(yoke_rvecs, yoke_tvecs)

    cv2.drawFrameAxes(frame, mtx, dist, yoke_rvec, yoke_tvec, 1)

    # move the yoke marker
    # tvectmp_cp = tvec[0] - camera_location
    yoke_rvec_b, yoke_tvec_b = relative_position(base_rvec, base_tvec, yoke_rvec, yoke_tvec)
    yoke_rvec_b2, yoke_tvec_b2 = relative_position( yoke_rvec, yoke_tvec,base_rvec, base_tvec)

    yoke_board_bb_o = sim.getObjectOrientation(yokeBoardCorner, baseBoardCorner)
    yoke_joint_w = sim.getJointPosition(yoke_joint1)
    sim.setJointPosition(yoke_joint1, yoke_rvec_b[2][0])
    yoke_board_c = sim.getObjectPosition(yokeBoardCorner, visionSensor)
    yoke_board_w = sim.getObjectPosition(yokeBoardCorner, -1)
    yoke_board_bb = sim.getObjectPosition(yokeBoardCorner, baseBoardCorner)

    yoke_board_position = [-yoke_tvec_b[0][0], -yoke_tvec_b[1][0],
                           -yoke_tvec_b[2][0]]

    sim.setObjectPosition(yokeBoardCorner, baseBoardCorner, yoke_board_position)

    sim.setJointPosition(joints[1], 0)
    sim.setJointPosition(joints[2], 0)
    sim.setObjectPosition(z1_robot, gripperBoardCorner, [0, 0, 0])
    gripper_board_corners, gripper_obj_points, gripper_img_points, gripper_board = \
        create_grid_board(config, aruco_dict, gray, corners, ids, mtx, dist, config.gripper, config.gripper + 1)

    gripper_flag, gripper_rvecs, gripper_tvecs, ggripper_r2 = cv2.solvePnPGeneric(
        gripper_obj_points, gripper_img_points, mtx, dist,
        flags=cv2.SOLVEPNP_IPPE)
    gripper_rvec, gripper_tvec = pick_rvec_board(gripper_rvecs, gripper_tvecs)
    cv2.drawFrameAxes(frame, mtx, dist, gripper_rvec, gripper_tvec, 1)

    # move the yoke marker
    # tvectmp_cp = tvec[0] - camera_location
    gripper_rvec_b, gripper_tvec_b = relative_position(base_rvec, base_tvec, gripper_rvec, gripper_tvec)

    gripper_rvec_inv, gripper_rvec_inv = invert_vec(gripper_rvec, gripper_tvec)

    gripper_board_c = sim.getObjectPosition(gripperBoardCorner, visionSensor)
    gripper_position = [-gripper_tvec[0][0], -gripper_tvec[1][0],
                        gripper_tvec[2][0]]
    sim.setObjectPosition(gripperBoardCorner, visionSensor, gripper_position)

    yoke_handle_w = sim.getObjectPosition(yoke_handle, -1)
    camera_w = sim.getObjectPosition(visionSensor, -1)

    tip_rb = sim.getObjectPosition(tip_world, base_world)
    yoke_rb = sim.getObjectPosition(yoke_world, base_world)
    yoke_rb_ori = sim.getObjectOrientation(yoke_world, base_world)
    yoke_rb_pose = sim.getObjectPose(yoke_world, base_world)
    yoke_world_90 = sim.getObject('/yoke_world_90')
    yoke_rb_pose_90 = sim.getObjectPose(yoke_world_90, base_world)

    # make sure all coordinates are in robot base so i can send them to the real robot
    sim.setObjectPose(target_handle, base_world, yoke_rb_pose_90)

    # ik_target = '/IK'
    # robotBaseHandle = sim.getObject(ik_target)
    # scriptHandle = sim.getScript(sim.scripttype_customizationscript, robotBaseHandle)
    # sim.callScriptFunction('handleIK', scriptHandle)

    # sim.handleVisionSensor(visionSensor)
    img, resX, resY = sim.getVisionSensorCharImage(visionSensor)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    corners_new, ids_new = detectArucoMarkers(img, parameters)
    cv2.aruco.drawDetectedMarkers(img, corners_new, ids_new)
    img_name = "image_{}.png".format(k)
    cv2.imwrite("test_video/" + img_name, img)

    if platform.system() != "Linux":
        continue
    # setup robot
    np.set_printoptions(precision=3, suppress=True)
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    ctrlComp = arm._ctrlComp
    udp = unitree_arm_interface.UDPPort(IP="127.0.0.1", toPort=8071, ownPort=8072)
    ctrlComp.udp = udp
    armModel = arm._ctrlComp.armModel
    joint_positions = []

    # Passive Mode and Calibration
    armState = unitree_arm_interface.ArmFSMState
    arm.loopOn()
    arm.setWait(True)
    arm.setFsm(armState.PASSIVE)
    arm.calibration()
    arm.loopOff()
    arm.setFsmLowcmd()

    for i in range(6):
        joint_positions.append(sim.getJointPosition(joints[i]))

    # gripper
    # joint_positions.append(0)
    duration = 1000
    lastPos = arm.lowstate.getQ()
    targetPos = np.array(joint_positions)  # forward

    for i in range(0, duration):
        arm.q = lastPos * (1 - i / duration) + targetPos * (i / duration)  # set position
        arm.qd = (targetPos - lastPos) / (duration * 0.002)  # set velocity
        arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6))  # set torque
        arm.gripperQ = -1 * (i / duration)
        arm.sendRecv()  # udp connection
        # print(arm.lowstate.getQ())
        time.sleep(arm._ctrlComp.dt)
    arm.loopOn()
    arm.backToStart()
    arm.loopOff()

    if setup.use_coppelia_sim:
        client.step()

    # show results on image
    if setup.show_image:
        cv2.namedWindow("Detection result", cv2.WINDOW_NORMAL)
        cv2.imshow("Detection result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # increment frame number
    k = k + config.frames.step

    # skip frames from video
    if config.use_video:
        for i in range(config.frames.step - 1):
            ret, frame = video.read()
            if ret == False:
                break

if config.use_video:
    video.release()

if setup.show_image:
    cv2.destroyAllWindows()
