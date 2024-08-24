import cv2
import numpy as np


class ArucoDetector(object):
    def __init__(self):  # Default dictionary is 4 for all 16x16 marker IDs up to 6 (8-bit ID)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)  # For 5x5 markers with IDs from 0-249
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
    def detect_and_draw_markers(self, image):
        """
        Detects aruco markers in the input image and draws them on the original image.

        Args:
            image: Input image to detect and draw markers from.

        Returns:
            image (numpy array): The modified frame with detected and drawn markers.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)

        # Draw detected markers
        for i, corner in enumerate(corners):
            cv2.drawContours(image, [corner.astype(np.int32)], -1, (0, 255, 0), 2)
        
        # Draw marker ID numbers
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # for i, id in enumerate(ids):
        #     x, y = corner[i][0] + corner[i][1] // 2, corner[i][3] + corner[i][4] // 2
        #     cv2.putText(image, str(id), (x, y), font, .5, (255, 0, 0), 2)

        return image
