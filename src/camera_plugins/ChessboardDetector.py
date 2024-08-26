import os
import time
import cv2
import numpy as np
from .Plugin import Plugin  # Import the base Plugin class

class ChessboardDetector(Plugin):
    """
    A camera plugin that detects chessboards in frames and displays a countdown timer.

    Attributes:
        chessboard_size (tuple): The number of inner corners per chessboard row and column.
            Defaults to (9, 6).
        countdown_time (int): The duration of the countdown timer in seconds. Defaults to 5.
        countdown_start_time (float or None): The time when the countdown started. Reset to None when the countdown ends.
    """

    def __init__(self):
        """
        Initializes the ChessboardDetector plugin.

        Sets default values for chessboard size and countdown time.
        """
        self.chessboard_size = (9, 6)  # Number of inner corners per chessboard row and column
        self.countdown_time = 5  # seconds
        self.countdown_start_time = None

    def run(self, frame, camID):
        """
        Runs the plugin on a single frame.

        Args:
            frame (numpy array): The input frame.
            camID (str or int): The ID of the camera that captured the frame.

        Returns:
            numpy array: The processed frame with detected chessboards and countdown timer displayed.
        """
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners in the grayscale frame
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        # Check if countdown has started
        if self.countdown_start_time is None:
            # If not, start the countdown and store the current time
            self.countdown_start_time = time.time()

        elapsed_time = time.time() - self.countdown_start_time

        # Display countdown on frame
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a font for displaying text
        height, width, _ = frame.shape  # Get the dimensions of the frame
        cv2.putText(frame, f"{int(self.countdown_time - elapsed_time + 1)}", (width // 2 - 40, height // 2), font, 4, (0, 255, 0), 4)  # Display countdown text

        # Check if countdown has ended
        if elapsed_time >= self.countdown_time:
            # If so, check if a chessboard was detected
            if ret:
                # Write frames to disk when a chessboard is detected
                save_dir = f"saved_images/{camID}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)  # Create the save directory if it doesn't exist
                filename = os.path.join(save_dir, f"{time.time()}.jpg")  # Generate a unique filename using timestamp
                cv2.imwrite(filename, frame)  # Save the frame to disk

            self.countdown_start_time = None  # Reset countdown

        # Draw chessboard corners on the frame if detected
        frame = cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)

        return frame