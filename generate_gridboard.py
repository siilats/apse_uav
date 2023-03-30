import cv2.aruco as aruco
import cv2
import numpy as np
# 1
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# 2
nx = 3
ny = 1

board = aruco.GridBoard((nx, ny),
                                   markerLength=0.05,
                                   markerSeparation=0.05,
                                   dictionary=aruco_dict, ids=np.arange(10, 13))
# 3
img = board.generateImage(outSize=(2000, 400), marginSize=50)
# img = cv2.aruco.drawPlanarBoard(board, outSize=(600, 600), marginSize=50)

# 4
cv2.imwrite("aruco_4x4.png", img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
