import cv2  # Import OpenCV library for computer vision tasks
import numpy as np  # Import NumPy library for numerical computations

from ..Plugin import Plugin  # Import the Plugin class from parent module


class ARImage(Plugin):
    """
    A plugin for detecting ARUCO markers in an image and overlaying them with a custom image.

    Attributes:
        aruco_dict (int): The dictionary used to create the ArUco marker.
        parameters (cv2.aruco.DetectorParameters): The detection parameters for the ArUco detector.
        detector (cv2.aruco.ArucoDetector): The ArUco detector instance.
        overlay_image (numpy.ndarray): The custom image to be overlaid on top of the ARUCO markers.

    Methods:
        __init__: Initializes the plugin and loads the overlay image.
        run: Runs the plugin on a given frame, detecting ARUCO markers and overlaying them with the custom image.
    """

    def __init__(self):
        """
        Initializes the plugin and loads the overlay image.

        Loads the pre-defined dictionary for 5x5 ArUco markers (IDs from 0-249), sets up the detection parameters,
        creates an instance of the ArUco detector, and loads the custom overlay image.
        """
        # Load the pre-defined dictionary for 5x5 ArUco markers
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        
        # Set up the detection parameters for the ArUco detector
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Create an instance of the ArUco detector using the dictionary and parameters
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Load the custom overlay image (e.g., a logo or icon)
        self.overlay_image = cv2.imread('./src/camera_plugins/aruco/test.png', cv2.IMREAD_UNCHANGED)

    def run(self, frame, camID):
        """
        Runs the plugin on a given frame, detecting ARUCO markers and overlaying them with the custom image.

        Args:
            frame (numpy.ndarray): The input frame to process.
            camID (int): The camera ID (not used in this implementation).

        Returns:
            numpy.ndarray: The processed frame with the ARUCO markers overlaid with the custom image.
        """
        # Convert the input frame from BGR to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers in the grayscale frame
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
