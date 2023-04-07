import dataclasses
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class BoardType(Enum):
    CHARUCO = 1
    ARUCO = 2


@dataclass
class BoardCoordinates:
    type: BoardType
    base_corners: np.ndarray
    base_ids: np.ndarray
    corners: tuple
    ids: np.ndarray
    base_board: cv2.aruco.CharucoBoard

    def is_ids_null_or_zero(self) -> bool:
        return np.all(self.ids == None) or len(self.ids) == 0

    def sort_corners(self) -> 'BoardCoordinates':
        ids = self.ids
        corners = self.corners
        idx = np.argsort(ids.ravel())
        corner_not_tuple = np.array(corners)[idx]
        corners = tuple(corner_not_tuple)
        ids = ids[idx]
        board_coordinates_copy = dataclasses.replace(self)
        board_coordinates_copy.corners = corners
        board_coordinates_copy.ids = ids
        return board_coordinates_copy
