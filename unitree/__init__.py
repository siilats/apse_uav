import cv2
from cv2 import aruco
import numpy as np
import json
import csv
import os
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import filedialog

# =======================================
# PARAMETERS TO BE CONFIGURED BY THE USER
# =======================================

# The index of the first frame to be processed.
start_frame = 1300

# The index of the last frame to be processed. If set to None, all frames from the input folder/input video folder
# will be processed. The processing can be immediately terminated by pressing the 'q' key.
stop_frame = 1339

# The number of frames to skip between each processed frame.
step_frame = 1

# If set to True, displays an image with the detection results.
showImage = True

# The value for the cv2.waitKey() function. If set to 0, the function will wait for a key to be pressed. Otherwise,
# it will wait for the specified time in milliseconds before showing the image.
cv2waitKeyVal = 1

# If set to True, the results will be saved to a file.
saveResults = False

# If set to True, the images after detection will be saved to disk.
saveImages = False

# If set to True, the data from the DCNN will be used in addition to the Aruco marker method.
useCentroidData = False

# The number of frames to use for averaging the marker size. Recommended value is 1.
N_avg = 1

# If set to True, the detected markers will be drawn on the image.
drawMarkers = True

# If set to True, the axes of the markers will be drawn on the image.
drawMarkersAxes = False

# If set to True, the pose and ID of the markers will be printed on the image.
showDataOnImage = True

# If set to True, the distances between vehicles will be printed on the image.
showDistancesOnImage = True

# If set to True, the LEDs of the host car will be drawn on the image.
drawLeds = False

# The threshold value for LED detection. If set to None, the default value of (190 + altitude in meters) will be used.
LEDs_threshold = None

# If set to True, the Lidar will be used as the source of measurements. If set to False, the host's Aruco marker will
# be used.
sourceLidar = False

# If set to True, lines will be drawn from the Lidar/host's Aruco marker to the vehicles. If set to False,
# no lines will be drawn. The color of the line will indicate the distance to the Aruco marker (red) and the distance
# to the closest point (yellow).
drawLines = True

# If set to True, points will be drawn on the image. The color of the points will indicate the Aruco centroid and
# Lidar (cyan), the DCNN centroid (magenta), and the DCNN closest point (white).
drawPoints = False

# Path to camera parameters file
path_camera_params = "data/cam_params.json"

# Ask user if they want to browse for paths
browse_paths = input("Would you like to browse for paths? (y/n): ").lower()

if browse_paths == "y":
    # Browse for input image folder
    root = tk.Tk()
    root.withdraw()
    path_input_images = filedialog.askdirectory(initialdir=".", title="Select Input Image Folder")

    # Browse for input video file
    path_input_video = filedialog.askopenfilename(initialdir=".", title="Select Input Video File",
                                                  filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi")))

    # Browse for DCNN data file
    path_dcnn_data = filedialog.askopenfilename(initialdir=".", title="Select DCNN Data File",
                                                filetypes=(("CSV files", "*.csv"),))

    # Browse for output results file
    path_output_results = filedialog.asksaveasfilename(initialdir=".", title="Save Output Results File",
                                                       filetypes=(("CSV files", "*.csv"),),
                                                       defaultextension=".csv")

    # Browse for output images folder
    path_output_images = filedialog.askdirectory(initialdir=".", title="Select Output Images Folder")

else:
    # Use default paths
    useImages = True
    path_input_images = "dynamic_images"
    useVideo = False
    path_input_video = "/Users/keithsiilats/Downloads/controltest.mp4"
    useCentroidData = False
    path_dcnn_data = "your_path"
    saveResults = False
    path_output_results = "your_path"
    saveImages = False
    path_output_images = "your_path"

# True if you use images as input, False if you use video
useImages = True

# True if you use video as input, False if you you images
useVideo = False

# path to data from DCNN detection, used only if useCentroidData is True (path + filename.csv)
useCentroidData = False

# path to save results to a file, used only if saveResults is True (path + filename.csv)
# be careful not to overwrite any existing file!
saveResults = False

# path to save images to a folder, used only if saveImages is True
# path must lead to an existing folder!
saveImages = False

# Professional comment explaining the purpose of the code
"""
This code allows the user to specify or browse for paths to input images or video, DCNN data, output results, and output
images. The user is prompted to browse for paths using a simple GUI, or use default paths if they choose not to browse. 
These paths are then used for processing input and generating output.
"""


# %%====================================
# FUNCTIONS FOR DATA INPUT/OUTPUT


def readCameraParams():
    """
    Reads the camera matrix and distortion coefficients from a JSON file.

    Returns:
    - mtx: a numpy array of shape (3,3) representing the camera matrix.
    - dist: a numpy array of shape (1,5) representing the distortion coefficients.
    """

    path = "data/cam_params.json"

    # check if the user wants to browse the path
    browse_path = input("Do you want to browse the camera parameters file path? (y/n)")
    if browse_path.lower() == "y":
        path = filedialog.askopenfilename(initialdir=".", title="Choose the Json File",
                                          filetypes=(("json files", "*.json"),))

    # read camera parameters from file
    with open(path, "r") as file:
        cam_params = json.load(file)

    # camera matrix
    mtx = np.array(cam_params["mtx"])

    # distortion coefficients
    dist = np.array(cam_params["dist"])

    return mtx, dist






# FUNCTIONS FOR SETTING PARAMETERS

def setArucoParameters():
    parameters = aruco.DetectorParameters()

    # set values for Aruco detection parameters
    parameters.minMarkerPerimeterRate = 0.01  # enables detection from higher altitude
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.33
    parameters.errorCorrectionRate = 2.0  # much more detections from high altitude, but FP happen sometimes
    parameters.aprilTagMinClusterPixels = 100  # less candidates to encode ID
    parameters.aprilTagMaxNmaxima = 5
    parameters.aprilTagCriticalRad = 20 * np.pi / 180  # much less candidates to encode ID
    parameters.aprilTagMaxLineFitMse = 1
    parameters.aprilTagMinWhiteBlackDiff = 100  # faster detection, but in bad contrast problems may happen
    # parameters.aprilTagQuadDecimate = 1.5 #huge detection time speedup, but at the cost of fewer detections and worse accuracy

    # default set of all Aruco detection parameters
    # parameters.adaptiveThreshWinSizeMin = 3
    # parameters.adaptiveThreshWinSizeMax = 23
    # parameters.adaptiveThreshWinSizeStep = 10
    # parameters.adaptiveThreshConstant = 7
    # parameters.minMarkerPerimeterRate = 0.03
    # parameters.maxMarkerPerimeterRate = 4
    # parameters.polygonalApproxAccuracyRate = 0.03
    # parameters.minCornerDistanceRate = 0.05
    # parameters.minDistanceToBorder = 3
    # parameters.minMarkerDistanceRate = 0.05
    # parameters.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    # parameters.cornerRefinementWinSize = 5
    # parameters.cornerRefinementMaxIterations = 30
    # parameters.cornerRefinementMinAccuracy = 0.1
    # parameters.markerBorderBits = 1
    # parameters.perspectiveRemovePixelPerCell = 4
    # parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    # parameters.maxErroneousBitsInBorderRate = 0.35
    # parameters.minOtsuStdDev = 5.0
    # parameters.errorCorrectionRate = 0.6
    # parameters.aprilTagMinClusterPixels = 5
    # parameters.aprilTagMaxNmaxima = 10
    # parameters.aprilTagCriticalRad = 10*np.pi/180
    # parameters.aprilTagMaxLineFitMse = 10
    # parameters.aprilTagMinWhiteBlackDiff = 5
    # parameters.aprilTagDeglitch = 0
    # parameters.aprilTagQuadDecimate = 0
    # parameters.aprilTagQuadSigma = 0
    # parameters.detectInvertedMarker = False

    return parameters


def setAverageMarkerSize():
    """
    Initializes and returns temp variables for averaging marker size.

    Returns:
    msp_avg -- tuple of numpy arrays of shape (N_avg, 1) representing the average marker size for each of the 4 markers.
    """

    # Initialize tuple to hold average marker sizes
    msp_avg = tuple(np.zeros((N_avg, 1)) for i in range(4))

    return msp_avg


# %%====================================
# FUNCTIONS FOR ARUCO MARKERS

def preprocessFrame(frame, mapx, mapy, lookUpTable):
    # Remove distortion from the camera using precomputed maps.
    # The maps are generated by cv2.initUndistortRectifyMap() and cv2.remap()
    # to correct for radial and tangential distortion.
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # Perform gamma correction using a Look Up Table (LUT).
    # The LUT maps pixel values to a new set of values, which can be used to apply
    # gamma correction (i.e., a nonlinear brightness adjustment) to the image.
    # Here, we apply gamma correction to the L channel of the LAB color space.
    # This helps to improve the contrast and brightness of the image.
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lookUpTable)

    # Convert back to RGB color space
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return frame


def detectArucoMarkers(gray, parameters):
    """
    Detect Aruco markers in an input grayscale image using the APRILTAG method.

    Parameters:
    gray (numpy.ndarray): Input grayscale image.
    parameters (cv2.aruco_DetectorParameters): Parameters for the Aruco detector.

    Returns:
    corners (list): List of detected marker corners.
    ids (numpy.ndarray): Array of detected marker IDs.
    """

    # Use a predefined Aruco markers dictionary.
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # Detect markers with APRILTAG method.
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    detector = aruco.ArucoDetector(aruco_dict)
    detector.setDetectorParameters(parameters)

    # Perform marker detection.
    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    return corners, ids


def getMarkerData(corners, rvec, cx_prev, cy_prev, markerLength):
    """
    Calculate marker center, size, distance to previous frame, and yaw angle.

    Args:
        corners: list of corner coordinates of the marker.
        rvec: rotation vector.
        cx_prev: x-coordinate of marker center in previous frame.
        cy_prev: y-coordinate of marker center in previous frame.

    Returns:
        A tuple containing the marker center x and y coordinates, marker size in pixels, distance
        between markers of the same ID on subsequent frames, and the yaw angle.
    """

    # Get the x and y coordinates of the marker center
    cx = 0
    cy = 0
    for corner in corners:
        cx += corner[0]
        cy += corner[1]
    cx /= 4
    cy /= 4

    # Calculate the size of the marker in pixels
    msp = 0
    for i in range(4):
        j = (i + 1) % 4
        msp += np.sqrt((corners[i][0] - corners[j][0]) ** 2 + (corners[i][1] - corners[j][1]) ** 2)
    msp /= 4

    # Calculate the distance between markers of the same ID on subsequent frames
    if cx_prev is not None and cy_prev is not None:
        diff = np.sqrt((cx_prev - cx) ** 2 + (cy_prev - cy) ** 2) * markerLength / msp
    else:
        diff = 0

    # Calculate the yaw angle
    r = R.from_rotvec(rvec)
    ang = r.as_euler('zxy', degrees=True)[0]

    return abs(cx), abs(cy), msp, diff, ang


def calculateAverageMarkerSize(msp_avg, msp):
    # write last measured marker size into table
    if (N_avg == 1):
        msp_avg = msp
    elif (N_avg > 1 and isinstance(N_avg, int)):
        for j in range(N_avg - 1):
            msp_avg[j] = msp_avg[j + 1]
        msp_avg[N_avg - 1] = msp

    # calculate the average and rescale marker size
    nonzero = np.count_nonzero(msp_avg)
    size_corr = np.sum(msp_avg) / (msp * nonzero)
    msp = msp * size_corr

    return size_corr, msp


def markerLengthCorrection(altitude, markerLengthOrg, marker_div, div ):
    # Calculates the correction factor for marker size based on the current altitude
    # The altitude is used to adjust the marker size as markers appear smaller at higher altitudes due to perspective
    # markerLengthOrg: original marker length
    # marker_div: a factor used to divide the altitude
    # div: a constant factor used in the calculation
    return markerLengthOrg * (1 - 0.00057 * altitude / marker_div) / div


def printDataOnImage(corners, tvec, rvec, ids, marker_div, frame):
    """
            Parameters:

        corners: A list of 4 corner points of a marker in an image. tvec: A translation vector that represents the
        distance between the marker and the camera. rvec: A rotation vector that represents the orientation of the
        marker in the camera frame. ids: An integer value that represents the ID of the marker. font_size: A float
        value that represents the size of the font used for displaying the text on the image. Default value is 1.4.
        font_thickness: An integer value that represents the thickness of the font used for displaying the text on
        the image. Default value is 2. Return: The function does not return anything. It only displays the text on
        the input image.

        Function Description: The printDataOnImage() function takes the corners, tvec, rvec, and ids of a marker in
        an image, along with optional parameters font_size and font_thickness, and displays the marker ID,
        real-world position, and Euler angles on the input image using the OpenCV library. The function converts the
        rotation vector to a rotation matrix, scales the z-coordinate of the translation vector by a marker_div
        constant, and calculates the real-world position and Euler angles using the rotation matrix. It then creates
        text for the ID, real-world position, and Euler angles, and defines the text positions relative to the marker
        corners. Finally, the function loops through each text and writes it onto the image using the putText()
        function of OpenCV.




    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    r = R.from_rotvec(rvec)

    # calculate real altitude to be printed
    tvec_temp = tvec.copy()
    tvec_temp[2] = tvec_temp[2] / marker_div

    # calculate angles and position and convert them to text
    ang = 'R = ' + str([round(r.as_euler('zxy', degrees=True)[0], 2),
                        round(r.as_euler('zxy', degrees=True)[1], 2),
                        round(r.as_euler('zxy', degrees=True)[2], 2)]) + 'deg'
    pos = 't = ' + str([round(j, 3) for j in tvec_temp]) + 'm'
    id = 'ID = ' + str(ids)

    # calculate the position where the text will be placed on image
    position = tuple([int(corners[0] - 150), int(corners[1] + 150)])
    position_ang = tuple([int(position[0] - 0), int(position[1] + 50)])
    position_id = tuple([int(position[0] - 0), int(position[1] - 50)])

    positions = [position_id, position, position_ang]
    ipa = [id, pos, ang]
    # write the text onto the image
    for i, pos in enumerate(positions):
        cv2.putText(frame, ipa[i], positions[i], font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)


# %%====================================
# FUNCTIONS FOR POINTS CALCULATIONS



def centroidFromAruco(veh_coords, tvec, rvec, size_corr, mtx, dist):
    # Project measured centroid of the vehicle wrt. Aruco marker onto image
    imgpts, _ = cv2.projectPoints(veh_coords, rvec, tvec / size_corr, mtx, dist)
    # Convert to integer and ensure non-negative values
    imgpts = np.maximum(0, np.int32(imgpts).reshape(-1, 2))

    # Draw the projected centroid if drawPoints is True
    if drawPoints:
        # Draw a filled circle with a radius of 5 pixels and color (255, 255, 0)
        cv2.circle(frame, (int(imgpts[0][0]), int(imgpts[0][1])), 5, color=(255, 255, 0), thickness=-1)

    # Return the projected centroid coordinates
    return imgpts


def centroidFromDCNN(centroid_data_x, centroid_data_y): ##Delete
    """
    Compute the centroid of the vehicle from DCNN detection.

    Args:
        centroid_data_x: x-coordinate of the centroid from DCNN detection.
        centroid_data_y: y-coordinate of the centroid from DCNN detection.

    Returns:
        An array containing the x and y coordinates of the centroid in the image.

    """
    # Extract x and y coordinates from input data
    xc = centroid_data_x
    yc = centroid_data_y

    # Compute image point from centroid coordinates
    imgpts = np.maximum(0, np.int32(np.array([[xc, yc, 0]])))

    # Draw point on the image if drawPoints flag is set
    if drawPoints:
        cv2.circle(frame, (int(imgpts[0][0]), int(imgpts[0][1])), 5, color=(255, 0, 255), thickness=-1)

    return imgpts




def drawBoundingBox(tvec, rvec, veh_dim, size_corr, mtx, dist, frame): #delete
    # Calculate angles in horizontal and vertical directions
    alpha_h = np.arctan(tvec[0] / tvec[2])
    alpha_v = np.arctan(tvec[1] / tvec[2])

    # Calculate yaw angle of the vehicle
    rotation = R.from_rotvec(rvec)
    yaw = round(rotation.as_euler('zxy', degrees=True)[0], 2)

    # If the yaw angle of the vehicle is negative, the alpha angles may be negative too
    alpha_h = alpha_h if yaw < 0 else -alpha_h
    alpha_v = alpha_v if yaw < 0 else -alpha_v

    # Modify dimensions of the vehicle's bounding box
    veh_dim = np.multiply(veh_dim, [1 - alpha_h / 2, 1 + alpha_h / 2, 1 - alpha_v / 2, 1 + alpha_v / 2])

    # Set corners of the bounding box based on modified values
    axis = np.float32([[veh_dim[2], veh_dim[0], 0], [veh_dim[2], veh_dim[1], 0], [veh_dim[3], veh_dim[1], 0],
                       [veh_dim[3], veh_dim[0], 0]])

    # Project the bounding box onto the image and draw it
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec / size_corr, mtx, dist)
    imgpts = np.maximum(0, np.int32(imgpts).reshape(-1, 2))
    cv2.drawContours(frame, [imgpts[0:4]], -1, (255, 0, 0), 5)

    return veh_dim


# %%====================================
# FUNCTIONS FOR DISTANCE CALCULATION
#delete
def generatePointsBoundingBox(veh_dim):
    # generate additional points on bounding box - 20 along the length and 8 along the width of the vehicle
    n_points_l = 20
    n_points_w = 8

    # generate points along the length and width of the bounding box
    points_l = np.linspace(veh_dim[0], veh_dim[1], n_points_l)
    points_w = np.linspace(veh_dim[2], veh_dim[3], n_points_w)

    # generate arrays to store the points along the four edges of the bounding box
    edge1 = np.zeros((n_points_l, 2))
    edge2 = np.zeros((n_points_l, 2))
    edge3 = np.zeros((n_points_w, 2))
    edge4 = np.zeros((n_points_w, 2))

    # set x and y values for the points along each edge
    edge1[:, 0] = points_l
    edge1[:, 1] = veh_dim[2]
    edge2[:, 0] = points_l
    edge2[:, 1] = veh_dim[3]
    edge3[:, 0] = veh_dim[0]
    edge3[:, 1] = points_w
    edge4[:, 0] = veh_dim[1]
    edge4[:, 1] = points_w

    # concatenate the points generated on each edge of bbox
    bbox = np.concatenate((edge1, edge2, edge3, edge4))

    # add a column of zeros to the array to represent the z-coordinate of the points
    bbox = np.hstack((bbox, np.zeros((bbox.shape[0], 1))))

    return bbox

#delete
def findMinimumDistanceBoundingBox(source, bbox, tvec, rvec, size_corr, mtx, dist):
    """
    Find the point on the generated bounding box that is closest to the source of the signal.

    Args:
        source (array): The coordinates of the source of the signal in the image.
        bbox (array): The coordinates of the generated bounding box.
        tvec (array): The translation vector of the camera.
        rvec (array): The rotation vector of the camera.
        size_corr (float): The correction factor for the size of the image.

    Returns:
        The coordinates of the point on the generated bounding box that is closest to the source of the signal.
    """

    # Project the generated bounding box points onto the image.
    imgpts, _ = cv2.projectPoints(bbox, rvec, tvec / size_corr, mtx, dist)
    imgpts = np.maximum(0, np.int32(imgpts).reshape(-1, 2))

    # Find the minimum distance between the source of the signal and the generated bounding box points.
    # Initialize the distance and index variables.
    distance = np.inf
    index = 0

    # Loop through all the generated bounding box points.
    for i in range(len(imgpts)):
        # Calculate the distance between the source and the current bounding box point.
        d = np.sqrt((source[0][0] - imgpts[i][0]) ** 2 + (source[0][1] - imgpts[i][1]) ** 2)

        # If the distance is less than the current minimum distance, update the minimum distance and index variables.
        if d < distance:
            distance = d
            index = i

    # Return the closest point.
    return imgpts[index]

#delete
def calculateDistance(lidar, aruco, bbox, markerLength, msp4, msp):
    # Calculate distances from LiDAR sensor to Aruco marker and vehicle's bounding box
    d_aruco = np.linalg.norm(lidar - aruco)
    d_bbox = np.linalg.norm(lidar - bbox)

    # Convert distances from pixels to meters using the average scale factor of marker length per pixel
    dist_aruco = d_aruco * markerLength / ((msp4 + msp) / 2)
    dist_bbox = d_bbox * markerLength / ((msp4 + msp) / 2)

    # Return the distances in meters
    return dist_aruco, dist_bbox
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


def drawLinesOnImage(frame, source, point, cx, cy, dist_aruco, angle, veh_id, ang1=0, ang4=0):
    """
    Draw lines on the input image to indicate distances and angles between the source of the measurement and the
    detected vehicle.

    Args:
        source (np.ndarray): The source of the measurement.
        point (np.ndarray): The closest point on the detected vehicle.
        cx (float): The x-coordinate of the center of the Aruco marker.
        cy (float): The y-coordinate of the center of the Aruco marker.
        dist_aruco (float): The distance between the source and the center of the Aruco marker.
        angle (float): The angle between the source-vehicle line and the horizontal axis.
        veh_id (str): The ID of the detected vehicle.
        ang1 (float): The angle of the detected vehicle's front left corner.
        ang4 (float): The angle of the detected vehicle's rear left corner.

    Returns:
        None
    """

    # Draw a line from the source of the measurement to the center of the detected vehicle's Aruco marker
    cv2.line(frame, (int(source[0][0]), int(source[0][1])), (int(cx), int(cy)), (0, 0, 255), 5)

    # If showDistancesOnImage is True, display the distances and angles on the image
    if showDistancesOnImage:
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Convert the distances and angles to text
        dist_aruco = str(round(dist_aruco, 1)) + ','
        angle = str(round(ang1 - ang4, 1)) + ' degrees'

        # Calculate the positions where the text will be placed on the image
        position_red = tuple([int((source[0][0] + cx) / 2 - 200), int((source[0][1] + cy) / 2) - 50])
        position_yellow = tuple([int((source[0][0] + cx) / 2 + 50), int((source[0][1] + cy) / 2) - 50])

        # Write the text onto the image
        cv2.putText(frame, dist_aruco, position_red, font, 3.0, (0, 0, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, angle, position_yellow, font, 3.0, (0, 255, 255), 6, cv2.LINE_AA)

