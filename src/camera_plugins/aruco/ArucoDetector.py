import cv2
import numpy as np
from ..Plugin import Plugin

class ArucoDetector(Plugin):
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)  # For 5x5 markers with IDs from 0-249
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
    def run(self, frame, camID):
        """
        Detects aruco markers in the input image and draws them on the original image.

        Args:
            image: Input image to detect and draw markers from.

        Returns:
            image (numpy array): The modified frame with detected and drawn markers.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)

        # Draw detected markers
        for i, corner in enumerate(corners):
            cv2.drawContours(frame, [corner.astype(np.int32)], -1, (0, 255, 0), 2)

        return frame
