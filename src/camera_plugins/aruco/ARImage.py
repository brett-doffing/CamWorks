import cv2
import numpy as np
from ..Plugin import Plugin

class ARImage(Plugin):
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)  # For 5x5 markers with IDs from 0-249
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        # Load the overlay image (e.g. a logo or icon)
        self.overlay_image = cv2.imread('./src/camera_plugins/aruco/test.png', cv2.IMREAD_UNCHANGED)

    def run(self, frame, camID):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)

        # Overlay the marker with an image
        for i, corner in enumerate(corners):
            # Get the bounding box of the marker
            x, y, w, h = cv2.boundingRect(corner.astype(np.int32))
            
            # Convert overlay image to RGBA format (if necessary)
            if self.overlay_image.shape[2] == 3:  # BGR format
                self.overlay_image = cv2.cvtColor(self.overlay_image, cv2.COLOR_BGR2BGRA)

            # Resize the overlay image to fit within the marker
            resized_overlay = cv2.resize(self.overlay_image, (w, h))
            
            # Overlay the resized image onto the original frame using the alpha channel
            bgr_resized_overlay = cv2.cvtColor(resized_overlay, cv2.COLOR_BGRA2BGR)
            result = cv2.addWeighted(bgr_resized_overlay, 1.0, frame[y:y+h, x:x+w], 0.0, 0)
            # Copy the overlaid region back into the original frame
            frame[y:y+h, x:x+w] = result

        return frame