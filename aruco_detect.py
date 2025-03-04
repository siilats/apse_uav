import os
import cv2
from cv2 import aruco
import numpy as np
import json
import csv
from scipy.spatial.transform import Rotation as R
from zmqRemoteApi import RemoteAPIClient
import argparse
import os
import time

class ArucoDetect:
    def __init__(self, ):
        #three versions, 1, original images with use_images
        #one with boards and the video with lamp
        #third is video of the plane
        #also has the flag for mirror to coppeliasim
        self.img_counter = 0

        # base_car = 1
        # moving_car = 4
        self.base_car = 1
        self.moving_car = 5
        self.use_boards = False
        self.use_coppelia_sim = True
        self.camera_location = None
        self.baseBoard_orientation = None
        self.floor_level = 0.01 #floor level in meters

        self.mtx, self.dist = None, None

        #True if you use images as input, False if you use video
        self.useImages = False


        #path to folder with input images
        #images inside must be named image_XXXX.png, where XXXX is the frame number
        if self.useImages:
            self.path_input_images = "dynamic_images"

        self.useVideo = not self.useImages
        #path to an input video (path + filename + extension)
        if self.useVideo:
            #path_input_video = "moving_cam.mp4" #"coppeliasim_video.mp4"
            self.path_input_video = "robot_on_plane.mp4"


        #%%====================================
        #PARAMETERS TO BE CHANGED BY USER

        #index of first frame to be processed
        self.start_frame = 1300 if not self.useVideo else 1
        #index of last frame to be processed, if None: all frames from input folder/input video folder will be processed
        #you can also terminate the processing immediately by press 'q' key
        #stop_frame = 1339
        self.stop_frame = self.start_frame + 39 if not self.useVideo else None
        #change the value if you want to skip some frames on the sequence
        self.step_frame = 1

        #True if you want to show image with results, False otherwise
        self.showImage = True
        #value for cv2.waitKey() function - 0: wait for key to be pressed, otherwise: time in miliseconds to show image
        self.cv2waitKeyVal = 1

        #True if you want to save the results to a file, False otherwise
        self.saveResults = False
        #True if you want to save images after detection on the disk, False otherwise
        self.saveImages = False

        #True if you use data from DCNN, False if you only use Aruco method
        self.useCentroidData = False

        #number of frames to be used for marker size averaging, recommended is 1
        self.N_avg = 1

        #True if you want to draw markers on image, False otherwise
        self.drawMarkers = True
        #True if you want to draw axes of the markers on image, False otherwise
        self.drawMarkersAxes = False
        #True if you want to print pose and ID of the markers on image, False otherwise
        self.showDataOnImage = True
        #True if you want to print distances between vehicles on image, False otherwise
        self.showDistancesOnImage = True

        #True if you want to draw LEDs of the host car, False otherwise
        self.drawLeds = False
        #threshold value for LEDs detection - None: use default value (190 + altitude in metres), 0-255: your value
        self.LEDs_threshold = None

        #True if you want Lidar to be the source of measurements, False if you want host's Aruco marker
        self.sourceLidar = False
        #True if you want to draw lines from Lidar/host's Aruco to vehicles, False otherwise
        #colour info: distance to Aruco marker - red, distance to closest point - yellow
        self.drawLines = True
        #True if you want to draw points on the image, False otherwise
        #colour info: Aruco centroid and Lidar - cyan, DCNN centroid - magenta, DCNN closest point - white
        self.drawPoints = False
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.frame = None

        #path to camera parameters file
        self.path_camera_params = "data/" + "cam_params.json"

        #path to data from DCNN detection, used only if useCentroidData is True (path + filename.csv)
        if self.useCentroidData:
            self.path_dcnn_data = "your_path"

        #path to save results to a file, used only if saveResults is True (path + filename.csv)
        #be careful not to overwrite any existing file!
        if self.saveResults:
            self.path_output_results = "your_path"

        #path to save images to a folder, used only if saveImages is True
        #path must lead to an existing folder!
        if self.saveImages:
            self.path_output_images = "your_path"



        #height, width = 2160, 3840 #fixed input image/video resolution
        self.height, self.width = None, None #fixed input image/video resolution
        if not self.use_boards:
            self.markerLengthOrg = 0.55 #real size of the marker in metres, this value does not change in algorithm
        else:
            self.markerLengthOrg = 0.05
            self.square_len = 0.064
            self.marker_seperation = 0.017
        self.markerLength = self.markerLengthOrg #real size of the marker in metres, this value changes in algorithm
        self.marker_div = 1.2 #correction for altitude estimation from marker
        self.div = 1.013 #additional correction for distance calculation (based on altitude test)
        self.DIFF_MAX = 2/3 * self.step_frame * 2 #maximum displacement of ArUco centre between frames with vehicle speed of 72 km/h = 20 m/s

        self.obj_points = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
        self.obj_points2 = np.array([[-self.markerLength / 2 , self.markerLength /  2, 0],
                    [self.markerLength / 2, self.markerLength / 2, 0],
                    [self.markerLength / 2, -self.markerLength / 2, 0],
                    [-self.markerLength / 2, -self.markerLength / 2, 0]])


        if self.use_coppelia_sim:
            self.client = RemoteAPIClient()
            self.sim = self.client.getObject('sim')
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                time.sleep(0.1)
            self.sim.loadScene(os.getcwd() + "/two_cars.ttt")
            self.visionSensorHandle = self.sim.getObject('/Vision_sensor')

            self.baseBoard = self.sim.getObject('/baseBoard')
            self.yokeBoard = self.sim.getObject('/yokeBoard')

            self.sim.setObjectPosition(self.baseBoard, -1, [0, 0, self.floor_level])
            self.sim.setObjectOrientation(self.baseBoard, -1, [180/360*2*3.1415, 0, -90/360*2*3.1415])
            self.sim.setObjectPosition(self.yokeBoard, -1, [10, 0, self.floor_level])
            self.sim.setObjectOrientation(self.yokeBoard, -1, [180/360*2*3.1415, 0, 60/360*2*3.1415])
            self.above_orientation = [-180/360*2*3.1415, 0, 180/360*2*3.1415]
            #sim.yawPitchRollToAlphaBetaGamma(visionSensorHandle, 180.0, 0.0, -180.0)
            #alpha, beta, gamma = sim.alphaBetaGammaToYawPitchRoll(-180/360*2*3.1415, 0, -180/360*2*3.1415)
            self.sim.setObjectOrientation(self.visionSensorHandle, -1, self.above_orientation)
            self.sim.setObjectPosition(self.visionSensorHandle, -1, [0, 0, 50])

            #get shapebb
            self.shapebb = self.sim.getShapeBB(self.baseBoard)
            self.shapebb_yoke = self.sim.getShapeBB(self.yokeBoard)
            self.coppeliasize = self.shapebb[0]
            self.targetsize = self.markerLengthOrg/100
            #set object scale
            #sim.scaleObject(baseBoard, targetsize/coppeliasize, targetsize/coppeliasize, targetsize/coppeliasize)
            #sim.scaleObject(yokeBoard, targetsize / shapebb_yoke[0], targetsize / shapebb_yoke[0], targetsize / shapebb_yoke[0])
            self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
            self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

            #sim.handleVisionSensor(visionSensorHandle)

            # Run a simulation in stepping mode:
            self.client.setStepping(True)
            self.sim.startSimulation()
            self.client.step()
            self.sim.addLog(self.sim.verbosity_scriptinfos, "all set up ---------------------------")

        if self.useCentroidData:
            self.centroid_data = self.readCentroidData(self.path_dcnn_data)  # read centroid data from DCNN
        if self.saveResults:
            self.file = self.outputDataInit()  # initialize output file for saving results

        self.parameters = self.setArucoParameters()  # create Aruco detection parameters
        # mtx, dist = readCameraParams() #read camera parameters
        self.msp1_avg, self.msp2_avg, self.msp3_avg, self.msp4_avg = self.setAverageMarkerSize()  # initialization of marker size averaging variables
        [self.cx1_prev, self.cy1_prev, self.cx2_prev, self.cy2_prev, self.cx3_prev, self.cy3_prev, self.cx4_prev,
         self.cy4_prev] = \
            np.zeros(8, dtype='int')  # initialization of ArUco marker centres

        self.gamma = 1  # gamma parameter value
        self.lookUpTable = np.empty((1, 256), np.uint8)  # look-up table for gamma correction
        for i in range(256):
            self.lookUpTable[0, i] = np.clip(pow(i / 255.0, self.gamma) * 255.0, 0, 255)

        # host vehicle's Lidar wrt. Aruco marker in metres
        self.veh4_coords_lidar = np.float32([[-0.05, -0.80, 0]])

        # vehicle's centroid wrt. Aruco marker in metres
        self.veh4_coords = np.float32([[0, 0.07, 0]])
        self.veh1_coords = np.float32([[0, 0.42, 0]])
        self.veh2_coords = np.float32([[0, 0.59, 0]])
        self.veh3_coords = np.float32([[0, 0.58, 0]])

        # initialize values if images are used
        if self.useImages:
            self.k = self.start_frame
            self.stop_frame = len(os.listdir(self.path_input_images)) if self.stop_frame is None else self.stop_frame
            self.frame = cv2.imread(self.path_input_images + "/image_%04d.png" % self.start_frame)

        # initialize values if video is used
        elif self.useVideo:
            self.video = cv2.VideoCapture(self.path_input_video)
            self.k = self.start_frame
            if self.start_frame > 0 and self.video.isOpened():
                for i in range(self.start_frame):
                    ret, self.frame = self.video.read()
                    if ret == False:
                        break
            self.stop_frame = np.inf if self.stop_frame is None else self.stop_frame

        self.height, self.width, self.channels = self.frame.shape

        self.fov = 60.0
        self.focal_length = self.width / (2 * np.tan(self.fov * np.pi / 360))
        self.mtx = np.array([[self.focal_length, 0, self.width / 2],
                        [0, self.focal_length, self.height / 2],
                        [0, 0, 1]])
        self.dist = None

        # calculate maps for undistortion
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, self.mtx, (self.width, self.height), 5)


    def readCameraParams(self):
        #read camera parameters from file
        with open(self.path_camera_params, "r") as file:
            self.cam_params = json.load(file)

        #camera matrix
        mtx = np.array(self.cam_params["mtx"])

        #distortion coefficients
        dist = np.array(self.cam_params["dist"])

        return mtx, dist

    def readCentroidData(self, path_dcnn_data):
        #open data file with centroids and bboxes from DCNN detection and store it in centroid_data variable
        self.centroid_data = []

        with open(path_dcnn_data) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 1:
                    temp = []
                    for i in range(17):
                        if row[i] == '' or row[i] == 'nan':
                            row[i] = 0
                        temp.append(int(row[i]))
                    centroid_data.append(temp)
                line_count += 1
        csv_file.close()

        return centroid_data

    def outputDataInit():
        #clear output file
        file = open(path_output_results, "w")

        #write names of the columns for data
        if useCentroidData:
            file.write("frame_ID ,ID_4_detected ,markerLength ,leds_ID ,UAV_altitude ,fov_width ,fov_height ," +
                    "ID_1_detected ,distance_veh1_aruco ,distance_veh1_aruco_bbox ,distance_veh1_dcnn ,distance_veh1_dcnn_bbox ," +
                    "ID_2_detected ,distance_veh2_aruco ,distance_veh2_aruco_bbox ,distance_veh2_dcnn ,distance_veh2_dcnn_bbox ," +
                    "ID_3_detected ,distance_veh3_aruco ,distance_veh3_aruco_bbox ,distance_veh3_dcnn ,distance_veh3_dcnn_bbox" + "\n")
        else:
            file.write("frame_ID ,ID_4_detected ,markerLength ,leds_ID ,UAV_altitude ,fov_width ,fov_height ," +
                    "ID_1_detected ,distance_veh1_aruco ,distance_veh1_aruco_bbox ," +
                    "ID_2_detected ,distance_veh2_aruco ,distance_veh2_aruco_bbox ," +
                    "ID_3_detected ,distance_veh3_aruco ,distance_veh3_aruco_bbox ," + "\n")

        file.close()
        file = open(path_output_results, "a")

        return file

    #%%====================================
    #FUNCTIONS FOR SETTING PARAMETERS

    def setArucoParameters(self):
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

        parameters.aprilTagMinWhiteBlackDiff = 10 #faster detection, but in bad contrast problems may happen
        #parameters.aprilTagQuadDecimate = 1.5 #huge detection time speedup, but at the cost of fewer detections and worse accuracy

        #default set of all Aruco detection parameters

        parameters.adaptiveThreshWinSizeMin = 50
        parameters.adaptiveThreshWinSizeMax = 400
        parameters.adaptiveThreshWinSizeStep = 40

        #parameters.useAruco3Detection = True

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

    def setAverageMarkerSize(self):
        #temp variables for averaging marker size
        msp1_avg = np.zeros((self.N_avg,1))
        msp2_avg = np.zeros((self.N_avg,1))
        msp3_avg = np.zeros((self.N_avg,1))
        msp4_avg = np.zeros((self.N_avg,1))

        return msp1_avg, msp2_avg, msp3_avg, msp4_avg

    #%%====================================
    #FUNCTIONS FOR ARUCO MARKERS

    def preprocessFrame(self, frame):
        #remove distortion from camera
        frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

        #perform gamma correction
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        lab[...,0] = cv2.LUT(lab[...,0], self.lookUpTable)
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return frame

    def detectArucoMarkers(self, gray, parameters):
        #use predefined Aruco markers dictionary

        #detect markers with APRILTAG method
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        detector = aruco.ArucoDetector(self.aruco_dict)
        detector.setDetectorParameters(parameters)

        corners, ids, rejected_img_points = detector.detectMarkers(gray)

        return corners, ids

    def getMarkerData(self, corners, rvec, cx_prev, cy_prev):
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
            diff = np.sqrt(np.power(cx_prev-cx,2) + np.power(cy_prev-cy,2)) * self.markerLength / msp
        else:
            diff = 0
        r = R.from_rotvec(rvec)
        ang = r.as_euler('zxy', degrees=True)[0]
        return abs(cx), abs(cy), msp, diff, ang

    def calculateAverageMarkerSize(self, msp_avg, msp):
        #write last measured marker size into table
        if(self.N_avg == 1):
            msp_avg = msp
        elif(self.N_avg > 1 and isinstance(self.N_avg, int)):
            for j in range(N_avg-1):
                msp_avg[j] = msp_avg[j+1]
            msp_avg[self.N_avg-1] = msp

        #calculate the average and rescale marker size
        nonzero = np.count_nonzero(msp_avg)
        size_corr = np.sum(msp_avg)/(msp*nonzero)
        msp = msp * size_corr

        return size_corr, msp

    def markerLengthCorrection(self, altitude):
        #use correction of marker size based on current altitude
        return self.markerLengthOrg * (1 - 0.00057 * altitude/self.marker_div) / self.div

    def printDataOnImage(self, corners, tvec, rvec, ids):
        font = cv2.FONT_HERSHEY_SIMPLEX
        r = R.from_rotvec(rvec)

        #calculate real altitude to be printed
        tvec_temp = tvec.copy()
        tvec_temp[2] = tvec_temp[2]/self.marker_div

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
        cv2.putText(self.frame, id, position_id, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.frame, pos, position, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.frame, ang, position_ang, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

    #%%====================================
    #FUNCTIONS FOR POINTS CALCULATIONS

    def detectAndDrawLEDs(self, gray, tvec, rvec, size_corr, msp, threshold = None):
        #position of the LEDs wrt. Aruco marker
        axis_leds = np.float32([[-0.419,-0.42,0],[-0.414,-0.305,0],[-0.409,-0.19,0],[-0.404,-0.07,0],
                                [-0.399,0.065,0],[-0.393,0.19,0],[-0.388,0.315,0],[-0.382,0.435,0]])

        #project these points onto the image
        imgpts_leds, _ = cv2.projectPoints(axis_leds, rvec, tvec/size_corr, self.mtx, self.dist)
        imgpts_leds = np.maximum(0,np.int32(imgpts_leds).reshape(-1,2))

        #use 190 + altitude in metres as the default value if the user did not specify the threshold
        thr = max(190 + int(tvec[2]/self.marker_div), 240) if threshold is None else threshold

        value = ''
        leds = 0
        for j in range(8):
            x = int(imgpts_leds[j][0])
            y = int(imgpts_leds[j][1])

            #use 5x5 neighbourhood of pixels
            point = gray[y-2:y+3,x-2:x+3]
            val = np.sum(np.sum(point))/25

            #if the LED is on
            if val > thr:
                value = value + '1'
                leds = leds + np.power(2,7-j)
                if drawLeds:
                    cv2.circle(self.frame, (x,y), int(msp/15)+1, color=(0,255,0), thickness=int(msp/30)+1)

            #if the LED is off
            else:
                value = value + '0'
                if self.drawLeds:
                    cv2.circle(self.frame, (x,y), int(msp/15)+1, color=(0,0,255), thickness=int(msp/30)+1)

        return leds

    def centroidFromAruco(self, veh_coords, tvec, rvec, size_corr):
        #project measured centroid of the vehicle wrt. Aruco marker onto image
        imgpts, _ = cv2.projectPoints(veh_coords, rvec, tvec/size_corr, self.mtx, self.dist)
        imgpts = np.maximum(0,np.int32(imgpts).reshape(-1,2))
        if self.drawPoints:
            cv2.circle(self.frame, (int(imgpts[0][0]),int(imgpts[0][1])), 5, color=(255,255,0), thickness=-1)

        return imgpts

    def centroidFromDCNN(self, centroid_data_x, centroid_data_y):
        #use the centroid of the vehicle from DCNN detection
        xc = centroid_data_x
        yc = centroid_data_y

        #set and draw the point on the image
        imgpts = np.maximum(0,np.int32(np.array([[xc, yc, 0]])))
        if self.drawPoints:
            cv2.circle(self.frame, (int(imgpts[0][0]),int(imgpts[0][1])), 5, color=(255,0,255), thickness=-1)

        return imgpts

    def boundingBoxFromDCNN(self, centroid_data_x, centroid_data_y):
        #use the closest point of the vehicle from DCNN detection
        xc = centroid_data_x
        yc = centroid_data_y
        imgpts = np.maximum(0,np.int32(np.array([[xc, yc, 0]])))
        if self.drawPoints:
            cv2.circle(self.frame, (int(imgpts[0][0]),int(imgpts[0][1])), 5, color=(255,255,255), thickness=-1)

        return imgpts

    def drawBoundingBox(self, tvec, rvec, veh_dim, size_corr):
        #calculate angles in horizontal and vertical direction
        alpha_h = np.arctan(tvec[0]/tvec[2])
        alpha_v = np.arctan(tvec[1]/tvec[2])

        #calucalate yaw angle of the vehicle
        r = R.from_rotvec(rvec)
        yaw = round(r.as_euler('zxy', degrees=True)[0],2)

        #based on yaw angle of the vehicle, alpha angles may be negative
        alpha_h = alpha_h if yaw < 0 else -alpha_h
        alpha_v = alpha_v if yaw < 0 else -alpha_v

        #modify dimensions of vehicle's bbox
        veh_dim = np.multiply(veh_dim, [1-alpha_h/2, 1+alpha_h/2, 1-alpha_v/2, 1+alpha_v/2])

        #use modified values to set corners of bbox, project these points onto the image and draw bbox
        axis = np.float32([[veh_dim[2],veh_dim[0],0], [veh_dim[2],veh_dim[1],0], [veh_dim[3],veh_dim[1],0], [veh_dim[3],veh_dim[0],0]])
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec/size_corr, self.mtx, self.dist)
        imgpts = np.maximum(0,np.int32(imgpts).reshape(-1,2))
        cv2.drawContours(self.frame, [imgpts[0:4]], -1, (255,0,0), 5)

        return veh_dim

    #%%====================================
    #FUNCTIONS FOR DISTANCE CALCULATION

    def generatePointsBoundingBox(self, veh_dim):
        #generate additional points on bounding box - 20 along the length and 8 along the width of the vehicle
        points_l = 20
        points_w = 8

        o1 = np.linspace(veh_dim[0], veh_dim[1], points_l)
        o2 = np.linspace(veh_dim[2], veh_dim[3], points_w)

        object1 = np.zeros((points_l,2))
        object2 = np.zeros((points_l,2))
        object3 = np.zeros((points_w,2))
        object4 = np.zeros((points_w,2))

        object1[:,0] = o1
        object1[:,1] = veh_dim[2]
        object2[:,0] = o1
        object2[:,1] = veh_dim[3]
        object3[:,0] = veh_dim[0]
        object3[:,1] = o2
        object4[:,0] = veh_dim[1]
        object4[:,1] = o2

        #concatenate the points generated on each edge of bbox
        object = np.concatenate((object1, object2, object3, object4))
        w, h = object.shape
        bbox = np.zeros((w, h+1))

        bbox[:,0] = object[:,1]
        bbox[:,1] = object[:,0]
        bbox[:,2] = 0

        return bbox

    def findMinimumDistanceBoundingBox(self, source, bbox, tvec, rvec, size_corr):
        #project generated bbox points onto image
        imgpts, _ = cv2.projectPoints(bbox, rvec, tvec/size_corr, self.mtx, self.dist)
        imgpts = np.maximum(0,np.int32(imgpts).reshape(-1,2))

        #find minimum distance between source of signal and generated bbox points
        distance = np.inf
        index = 0
        for i in range(len(imgpts)):
            d = np.sqrt(pow(source[0][0]-imgpts[i][0],2) + pow(source[0][1]-imgpts[i][1],2))
            if(d < distance):
                distance = d
                index = i

        #return the closest point
        return imgpts[index]

    def calculateDistance(self, lidar, aruco, bbox, markerLength, msp4, msp):
        #calculate distances to Aruco marker and bbox of the vehicle
        d_aruco = np.sqrt((lidar[0][0]-aruco[0][0]) * (lidar[0][0]-aruco[0][0]) + (lidar[0][1]-aruco[0][1]) * (lidar[0][1]-aruco[0][1]))
        d_bbox = np.sqrt((lidar[0][0]-bbox[0][0]) * (lidar[0][0]-bbox[0][0]) + (lidar[0][1]-bbox[0][1]) * (lidar[0][1]-bbox[0][1]))

        #convert distances from pixels to metres
        dist_aruco = d_aruco * markerLength / ((msp4+msp)/2)
        dist_bbox = d_bbox * markerLength / ((msp4+msp)/2)

        return dist_aruco, dist_bbox

    def drawLinesOnImage(self, source, point, cx, cy, dist_aruco, angle, veh_id, ang1=0, ang4=0):
        #draw the line from source of the measurement to the closest point of the vehicle
        cv2.line(self.frame, (int(source[0][0]), int(source[0][1])), (int(point[0]), int(point[1])), (0,255,255), 5)

        #draw the line from source of the measurement to the centre of vehicle' Aruco marker
        cv2.line(self.frame, (int(source[0][0]), int(source[0][1])), (int(cx), int(cy)), (0,0,255), 5)

        if self.showDistancesOnImage:
            font = cv2.FONT_HERSHEY_SIMPLEX

            #calculate angles and position and convert them to text
            dist_aruco = str(round(dist_aruco,1)) + ','
            angle = str(round(ang1 - ang4, 1)) + ' degrees'

            #calculate the position where the text will be placed on image
            position_red = tuple([int((source[0][0]+cx)/2-200), int((source[0][1]+cy)/2)-50])
            position_yellow = tuple([int((source[0][0]+cx)/2+50), int((source[0][1]+cy)/2)-50])

            #write the text onto the image
            cv2.putText(self.frame, dist_aruco, position_red, font, 3.0, (0, 0, 255), 6, cv2.LINE_AA)
            cv2.putText(self.frame, angle, position_yellow, font, 3.0, (0, 255, 255), 6, cv2.LINE_AA)


    def convert_angles(self, rvec):
        r = R.from_rotvec(rvec)
        ang = r.as_euler('zxy', degrees=True)
        return ang


    def pick_rvec(self, rvecs, tvecs):
        generic_ang1 = self.convert_angles(rvecs[0].ravel())
        generic_ang2 = self.convert_angles(rvecs[1].ravel())
        if abs(generic_ang1[2] - 180) < abs(generic_ang2[2] - 180):
            rvectmp = rvecs[0].ravel()
            tvectmp = tvecs[0].ravel()
        else:
            rvectmp = rvecs[1].ravel()
            tvectmp = tvecs[1].ravel()

        return rvectmp, tvectmp


    def matchImagePointsforcharuco(self, charuco_corners, charuco_ids):
        base_obj_pts = []
        base_img_pts = []
        for i in range(0, len(charuco_ids)):
            index = charuco_ids[i]
            base_obj_pts.append(self.base_board.getChessboardCorners()[index])
            base_img_pts.append(charuco_corners[i])

        base_obj_pts = np.array(base_obj_pts)
        base_img_pts = np.array(base_img_pts)

        return base_obj_pts, base_img_pts


    def run(self):
        moving_car_detected = 0
        base_car_detected = 0
        cx1,cy1,cx4,cy4 = 0,0,0,0
        msp4,msp1,msp=0,0,0
        ang1,ang4=0,0
        while self.k <= self.stop_frame and (self.useImages or (self.useVideo and self.video.isOpened())):
            #read frame from image or video
            if self.useImages:
                self.frame = cv2.imread(self.path_input_images + "/image_%04d.png" % k)
            elif self.useVideo:
                ret, self.frame = self.video.read()
                if ret == False:
                    break

            self.height, self.width, _ = self.frame.shape

            #real vehicle dimensions in metres wrt. Aruco marker: back, front, left, right
            veh4_dim = [-2.35, 2.49, -0.86, 0.86]
            veh1_dim = [-1.95, 2.8, -0.9, 0.9]
            veh2_dim = [-1.68, 2.86, -0.87, 0.87]
            veh3_dim = [-1.32, 2.48, -0.86, 0.86]

            #frame preprocessing - camera distortion removal and gamma correction
            self.frame = self.preprocessFrame(self.frame)

            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            # gray = cv2.medianBlur(gray, 3)
            # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 5)
            # gray = gray.astype(np.uint8)

            if self.use_boards:
                base_board_size = (3, 3)
                marker_len = self.markerLengthOrg
                square_len = 0.064
                marker_seperation = 0.017
                base_board = aruco.CharucoBoard(base_board_size, squareLength=square_len,
                                                markerLength=marker_len, dictionary=self.aruco_dict,
                                                ids=np.arange(4))
                base_detector = aruco.CharucoDetector(base_board)
                base_detector.setDetectorParameters(self.parameters)
                # define the planar aruco board and its detector
                # the default ids is np.arange(24, 27)
                base_corners, base_ids, corners, ids = \
                    base_detector.detectBoard(gray)

            else:
                corners, ids = self.detectArucoMarkers(gray, self.parameters)
            if (ids is not None) and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(self.frame, corners, ids)
            # write me adaptive grayscale in opencv
            idx = np.argsort(ids.ravel())
            corner_not_tuple = np.array(corners)[idx]
            corners = tuple(np.array(corners)[idx])
            ids = ids[idx]

            #%%====================================
            #MARKER DETECTION AND POINTS CALCULATIONS
            tvec = None
            rvec = None
            #if any marker was detected
            if np.all(ids != None):
                #estimate pose of detected markers
                if self.use_boards:
                    yoke_board_size = (3, 1)

                    yoke_board = aruco.GridBoard(yoke_board_size, markerLength=marker_len,
                                                      markerSeparation=marker_seperation, dictionary=aruco_dict,
                                                      ids=np.arange(24, 27))

                    yoke_detector = aruco.ArucoDetector(dictionary=aruco_dict,
                                                             refineParams=aruco.RefineParameters())
                    # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
                    ids_for_planar_board = ids.ravel() > 10
                    yoke_board_corners, yoke_board_ids, yoke_rejectedCorners, yoke_recoveredIdxs = \
                    yoke_detector.refineDetectedMarkers(gray, yoke_board,
                                                                  np.asarray(corners)[ids_for_planar_board],
                                                                  ids[ids_for_planar_board].ravel(),
                                                                cameraMatrix=mtx,
                                                                rejectedCorners=None,
                                                                distCoeffs=dist)

                    yoke_obj_points, yoke_img_points = yoke_board.matchImagePoints(yoke_board_corners, yoke_board_ids)
                    yoke_flag, yoke_rvecs, yoke_tvecs, yoke_r2 = cv2.solvePnPGeneric(
                        yoke_obj_points,yoke_img_points, mtx, dist,
                        flags=cv2.SOLVEPNP_IPPE)
                    rvectmp, tvectmp = self.pick_rvec(yoke_rvecs, yoke_tvecs)
                    tvec[0] = tvectmp
                    rvec[0] = rvectmp

                    base_obj_points, base_img_points = self.matchImagePointsforcharuco(base_corners, base_ids)

                    # bug in cv.solvePnPGeneric
                    base_flag, base_rvecs, base_tvecs, base_reproj_error = cv2.solvePnPGeneric(base_obj_points,
                                                                     base_img_points,
                                                                     self.mtx,
                                                                     self.dist,
                                                                     flags=cv2.SOLVEPNP_IPPE)

                    rvectmp, tvectmp = self.pick_rvec(base_rvecs, base_tvecs)
                    tvec[3] = tvectmp
                    rvec[3] = rvectmp
                    ids[3][0] = 4
                    ids[2][0] = 0
                    ids[1][0] = 0
                    ids[0][0] = 1
                    moving_car_detected=1
                    base_car_detected=1

                    cx1, cy1, msp1, diff1, ang1 = self.getMarkerData(base_corners.squeeze(), rvec[0], None if k == self.start_frame else self.cx1_prev,
                                                               None if k == self.start_frame else self.cy1_prev)  # get detected marker parameters

                    cx4, cy4, msp4, diff4, ang4 = self.getMarkerData(yoke_board_corners[0].squeeze(), rvec[3], None if k == self.start_frame else self.cx4_prev,
                                                               None if k == self.start_frame else self.cy4_prev)  # get detected marker parameters

                    size_corr1, msp1 = self.calculateAverageMarkerSize(msp1_avg, msp1)  # marker size averaging
                    size_corr4, msp4 = self.calculateAverageMarkerSize(msp4_avg, msp4)  # marker size averaging

                else:
                    #iterate over all detected markers
                    for i in range(len(ids)):
                        #only markers with ID={1,2,3,4} are used at this moment
                        # rvectmp=rvec[i][0] #compartible w previous version
                        # tvectmp=tvec[i][0] #compartible w previous version
                        flag, rvecs, tvecs, r2 = cv2.solvePnPGeneric(
                            self.obj_points2, corners[i], self.mtx, self.dist,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE)

                        rvectmp, tvectmp = self.pick_rvec(rvecs, tvecs)

                        tvec[i] = tvectmp
                        rvec[i] = rvectmp

                        if ids[i][0] == self.moving_car: #vehicle 4 (host)
                            cx4, cy4, msp, diff4, ang4 = self.getMarkerData(corners[i][0], rvectmp, None if
                                                self.k == self.start_frame else self.cx4_prev, None if self.k == self.start_frame else
                                                                    self.cy4_prev) #get detected marker parameters

                            if self.use_coppelia_sim and self.camera_location is not None:
                                # move the yoke marker
                                tvectmp_cp = tvectmp.copy() - self.camera_location
                                tvectmp_cp[2] = self.floor_level

                                #tvectmp_cp[1] = -tvectmp_cp[1]
                                self.sim.setObjectPosition(self.yokeBoard, -1, [tvectmp_cp[0], -tvectmp_cp[1], tvectmp_cp[2]])
                                print("moved yokeBoard", tvectmp_cp.tolist())

                                r4 = R.from_rotvec(rvectmp)
                                #rvectmp[1] = -rvectmp[1]
                                ang4 = r4.as_euler('zxy', degrees=True)[0]
                                correction = -90 - self.baseBoard_orientation #-7
                                final_angle = 180 - ang4 + correction

                                #we have -90 and then angle4 is 120 and we want answe 45
                                self.sim.setObjectOrientation(self.yokeBoard, -1, [180/360*2*3.1415, 0, final_angle/360*2*3.1415])
                                self.client.step()

                                img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
                                img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
                                img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
                                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                                #create image name with img_counter padded with zeros

                                img_name = "image_{}.png".format(self.start_frame + self.img_counter)
                                self.img_counter += 1
                                #save image
                                corners_t, ids_t = self.detectArucoMarkers(img, self.parameters)
                                cv2.imwrite("test_video/" + img_name, img)

                            # if this marker was detected on previous frame and its position in the image is similar
                            if (moving_car_detected == 1 and diff4 < self.DIFF_MAX) or self.k == self.start_frame:
                                if self.drawMarkers:
                                    cv2.drawContours(self.frame, [np.maximum(0,np.int32(corners[i][0]))], -1, (0,255,0), 3)
                                if self.drawMarkersAxes:
                                    aruco.drawAxis(self.frame, self.mtx, self.dist, rvectmp, tvectmp, markerLength)
                                if self.showDataOnImage:
                                    self.printDataOnImage(corners[i][0][0], tvectmp, rvectmp, ids[i][0])

                                moving_car_detected = 1 #mark vehicle as detected
                                altitude = tvectmp[2] #altitude info
                                self.markerLength = self.markerLengthCorrection(altitude) #correction of original marker size based on altitude
                                altitude = altitude/self.marker_div #calculate real altitude

                                size_corr4, self.msp4 = self.calculateAverageMarkerSize(self.msp4_avg, msp) #marker size averaging
                                leds = self.detectAndDrawLEDs(gray, tvectmp, rvectmp, size_corr4, msp4, self.LEDs_threshold) #LEDs detection

                                imgpts_veh4 = self.centroidFromAruco(self.veh4_coords, tvectmp, rvectmp, size_corr4) #calculate centroid of the vehicle wrt. Aruco marker
                                imgpts_veh4_lidar = self.centroidFromAruco(self.veh4_coords_lidar, tvectmp, rvectmp, size_corr4) #calculate Lidar's position wrt. Aruco marker
                                self.cx4_prev, self.cy4_prev = cx4, cy4 #save position of the marker in the image

                                if self.useCentroidData:
                                    imgpts_veh4_dcnn = self.centroidFromDCNN(self.centroid_data[k-1][1], self.centroid_data[k-1][2]) #calculate Aruco position wrt. vehicle centroid from DCNN
                                veh4_dim = self.drawBoundingBox(tvectmp, rvectmp, veh4_dim, size_corr4) #draw bounding box of the vehicle
                            else: #detected marker is a FP, change its ID to incorrect value
                                ids[i][0] = -1

                        if([self.base_car] not in ids): #if host is not detected, use altitude data from another marker
                            altitude = tvectmp[2] #altitude info
                            markerLength = self.markerLengthCorrection(altitude) #correction of original marker size based on altitude
                            altitude = altitude/self.marker_div #calculate real altitude

                        if(ids[i][0] == self.base_car): #vehicle 1
                            # get detected marker parameters
                            cx1, cy1, msp, diff1, ang1 = self.getMarkerData(corners[i][0], rvectmp, None if
                                                    self.k == self.start_frame else cx1_prev, None if
                                                                            self.k == self.start_frame else cy1_prev)

                            if self.use_coppelia_sim:
                                camera_orientation = rvectmp.copy()
                                camera_location = tvectmp

                            if base_car_detected == 0: #if this marker was not detected on previous frame, it may be 'new' or FP
                                base_car_detected = 1 #mark vehicle as detected
                                cx1_prev, cy1_prev = cx1, cy1 #save position of the marker in the image

                                if self.use_coppelia_sim:
                                    r1 = R.from_rotvec(camera_orientation)
                                    baseBoard_orientation = r1.as_euler('zxy', degrees=True)[0]
                                    camera_location_coppelia = [-camera_location[0], 0, camera_location[2]]
                                    # print("position: ", sim.getObjectPosition(visionSensorHandle, -1))
                                    # print("Camera location: ", camera_location)
                                    self.sim.setObjectPosition(self.visionSensorHandle, -1, camera_location_coppelia)
                                    #sim.addLog(sim.verbosity_scriptinfos, "all set up ---------------------------")


                            if (base_car_detected == 1 and diff1 < self.DIFF_MAX) or self.k == self.start_frame: #if this marker was detected on previous frame and its position in the image is similar
                                if self.drawMarkers:
                                    cv2.drawContours(self.frame, [np.maximum(0,np.int32(corners[i][0]))], -1, (0,255,0), 3)
                                if self.drawMarkersAxes:
                                    aruco.drawAxis(self.frame, self.mtx, self.dist, rvectmp, tvectmp, markerLength)
                                if self.showDataOnImage:
                                    self.printDataOnImage(corners[i][0][0], tvectmp, rvectmp, ids[i][0])

                                base_car_detected = 1 #mark vehicle as detected
                                size_corr1, msp1 = self.calculateAverageMarkerSize(self.msp1_avg, msp) #marker size averaging
                                imgpts_veh1 = self.centroidFromAruco(self.veh1_coords, tvectmp, rvectmp, size_corr1) #calculate centroid of the vehicle wrt. Aruco marker
                                cx1_prev, cy1_prev = cx1, cy1 #save position of the marker in the image

                                if self.useCentroidData:
                                    imgpts_veh1_dcnn = self.centroidFromDCNN(self.centroid_data[k-1][5], self.centroid_data[k-1][6]) #calculate Aruco position wrt. vehicle centroid from DCNN
                                    imgpts_veh1_dcnn_bbox = self.boundingBoxFromDCNN(self.centroid_data[k-1][7], self.centroid_data[k-1][8]) #calculate closest point of the vehicle from DCNN
                                veh1_dim = self.drawBoundingBox(tvectmp, rvectmp, veh1_dim, size_corr1) #draw bounding box of the vehicle
                            else: #detected marker is a FP, change its ID to incorrect value
                                ids[i][0] = -1

                        if self.use_coppelia_sim:
                            self.client.step()

                #%%====================================
                #DISTANCE CALCULATION FOR VEHICLES

                #iterate again over all detected markers to use results from current frame
                for i in range(len(ids)):
                    if(ids[i][0] == self.base_car): #get host car marker ID
                        #iterate over all markers to calculate distances to them from host
                        for j in range(len(ids)):
                            if(ids[j][0] == self.base_car): #vehicle 1
                                #start = time.time_ns()
                                if (moving_car_detected == 1 and diff1 < self.DIFF_MAX) or self.k == self.start_frame: #if this marker was detected on previous frame and its position in the image is similar
                                    bbox = self.generatePointsBoundingBox(veh1_dim) #generate additional points for bounding box
                                    if self.sourceLidar:
                                        point = self.findMinimumDistanceBoundingBox(imgpts_veh4_lidar, bbox, tvec[j], rvec[j], size_corr1) #find the closest point of the bbox from Lidar
                                        dist_veh1_aruco, dist_veh1_aruco_bbox = self.calculateDistance(imgpts_veh4_lidar, np.float32([[cx1, cy1]]), [point], markerLength, msp4, msp1) #calculate distances in metres for Aruco method
                                        if self.drawLines:
                                            self.drawLinesOnImage(imgpts_veh4_lidar, point, cx1, cy1, dist_veh1_aruco, dist_veh1_aruco_bbox, ids[j][0]) #draw lines between Lidar and vehicle
                                    else:
                                        point = self.findMinimumDistanceBoundingBox(np.float32([[cx4, cy4]]), bbox, tvec[j], rvec[j], size_corr1) #find the closest point of the bbox from host's Aruco
                                        dist_veh1_aruco, dist_veh1_aruco_bbox = self.calculateDistance(np.float32([[cx4, cy4]]), np.float32([[cx1, cy1]]), [point], self.markerLength, msp4, msp1) #calculate distances in metres for Aruco method
                                        if self.drawLines:
                                            self.drawLinesOnImage(np.float32([[cx4, cy4]]), point, cx1, cy1, dist_veh1_aruco, dist_veh1_aruco_bbox, ids[j][0], ang1, ang4) #draw lines between host's Aruco and vehicle
                                    if self.useCentroidData:
                                        dist_veh1_dcnn, dist_veh1_dcnn_bbox = self.calculateDistance(imgpts_veh4_lidar, imgpts_veh1_dcnn, imgpts_veh1_dcnn_bbox, markerLength, msp4, msp1) #calculate distances in metres for DCNN method

                self.cx4_prev, self.cy4_prev = cx4, cy4  # save position of the marker in the image
                self.cx1_prev, self.cy1_prev = cx1, cy1  # save position of the marker in the image


            #show results on image
            if self.showImage:
                cv2.namedWindow("Detection result", cv2.WINDOW_NORMAL)
                #resize image to fit screen
                cv2.resizeWindow("Detection result", 1280, 720)
                cv2.imshow("Detection result", self.frame)
                if cv2.waitKey(self.cv2waitKeyVal) & 0xFF == ord('q'):
                    break

            #save results to a file
            if self.saveResults:
                self.outputData(self.file)

            #save images to a folder
            if self.saveImages:
                cv2.imwrite(self.path_output_images + "image_%04d.png" % k, self.frame)

            #increment frame number
            self.k = self.k + self.step_frame

            #skip frames from video
            if self.useVideo:
                for i in range(self.step_frame-1):
                    ret, self.frame = self.video.read()
                    self.height, self.width, _ = self.frame.shape

                    if ret == False:
                        break

        if self.saveResults:
            self.file.close()

        if self.useVideo:
            self.video.release()

        if self.showImage:
            cv2.destroyAllWindows()

        if self.use_coppelia_sim:
            self.sim.stopSimulation()

            # Restore the original idle loop frequency:
            self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Example script with command line arguments.')
    parser.add_argument('--arg1', type=int, help='value of arg1')
    parser.add_argument('--arg2', type=int, help='value of arg2')
    args = parser.parse_args()

    aruco_detection = ArucoDetect()
    aruco_detection.run()