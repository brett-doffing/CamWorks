import cv2
import numpy as np
from ..Plugin import Plugin

class ArucoDetector(Plugin):
    """
    A plugin class for detecting Aruco markers in images.

    Attributes:
        aruco_dict (cv2.aruco.Dictionary): The dictionary of predefined Aruco marker IDs.
        parameters (cv2.aruco.DetectorParameters): The detector parameters for Aruco markers.
        detector (cv2.aruco.ArucoDetector): The Aruco detector instance.
    """

    def __init__(self):
        """
        Initializes the Aruco Detector plugin.

        In this method, we create an instance of `ArucoDetector` with a predefined dictionary
        of 5x5 Aruco markers and default detection parameters. This allows us to detect markers
        in images and draw them on the original image.
        """
        # Create a dictionary for 5x5 Aruco markers with IDs from 0-249
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        
        # Initialize detector parameters for Aruco markers
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Create an instance of the Aruco detector with the predefined dictionary and parameters
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def run(self, frame, camID):
        """
        Detects Aruco markers in the input image and draws them on the original image.

        Args:
            frame (numpy array): The input image to detect and draw markers from.
            camID (int or str): The camera ID associated with the input image.

        Returns:
            image (numpy array): The modified frame with detected and drawn markers.

        Notes:
            This method first converts the input image to grayscale, then uses the Aruco detector
            to find corners of marker candidates in the image. It then draws these markers on the
            original image.
        """
        # Convert the input image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Aruco markers in the grayscale image and retrieve their IDs
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)

        # Draw detected markers on the original image
        for i, corner in enumerate(corners):
            # Draw a green contour around each marker
            cv2.drawContours(frame, [corner.astype(np.int32)], -1, (0, 255, 0), 2)

        return frame
