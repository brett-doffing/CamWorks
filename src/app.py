import os
from Camera import Camera
import cv2
import numpy as np
from SiftManager import SiftManager
from dotenv import load_dotenv

load_dotenv()


def main():
    cameras = []
    ADDRESSES = os.getenv("ADDRESSES").split(" ")
    CAM_IDS = os.getenv("CAM_IDS").split(" ")
    grid_image = np.zeros((480 * 2, 640 * 2, 3), dtype=np.uint8)
    frames_dict = {}
    sift_manager = SiftManager()

    for addr, camid in zip(ADDRESSES, CAM_IDS):
        camera = Camera(addr, camid)
        cameras.append(camera)

    while True:
        # Fetch the latest frame from each camera
        for i, camera in enumerate(cameras):
            frame = camera.get_latest_frame()
            if frame is not None:
                frames_dict[CAM_IDS[i]] = frame  # Store latest frame in dictionary
        
        # Get the latest frames from the dictionary
        frame1 = frames_dict.get(CAM_IDS[0], np.zeros((480, 640, 3), dtype=np.uint8))
        frame2 = frames_dict.get(CAM_IDS[1], np.zeros((480, 640, 3), dtype=np.uint8))
        frame3 = frames_dict.get(CAM_IDS[2], np.zeros((480, 640, 3), dtype=np.uint8))
        frame4 = frames_dict.get(CAM_IDS[3], np.zeros((480, 640, 3), dtype=np.uint8))

        # # Assign the frames to the grid image without padding
        grid_image[:480, :640] = frame1  # Top-left
        grid_image[:480, 640:1280] = frame2  # Top-right
        grid_image[480:960, :640] = frame3  # Bottom-left
        grid_image[480:960, 640:1280] = frame4  # Bottom-right
        
        cv2.imshow('Camera Feed', grid_image)
        
        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Perform SIFT algorithm
            sift_manager.add_frame(CAM_IDS[0], frame1)
            sift_manager.add_frame(CAM_IDS[1], frame2)
            sift_manager.add_frame(CAM_IDS[2], frame3)
            sift_manager.add_frame(CAM_IDS[3], frame4)
            sift_manager.process_frames()

    # Stop all camera instances and close windows
    for camera in cameras:
        camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
