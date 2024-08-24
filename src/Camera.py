import zmq
import threading
import queue
import cv2
import numpy as np
from ArucoDetector import ArucoDetector  # Import the ArucoDetector class

class Camera(object):
    def __init__(self, address, camID):
        self._context = zmq.Context()
        self.arucoDetector = ArucoDetector()
        try:
            self.address = address
            self.camID = camID
            # Debug print for initialization
            self._socket = self.create_socket(self._context, self.address)
            self._frame_queue = queue.Queue()  # Queue for thread-safe frame updates
            self._running = threading.Event()
            self._thread = threading.Thread(target=self.receiveFeed, args=(self._socket,))
            self._thread.start()
        except Exception as e:
            print(f"Failed to initialize Camera: {e}")
    
    def create_socket(self, context, address):
        try:
            socket = context.socket(zmq.SUB)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.setsockopt(zmq.CONFLATE, 1) #NOTE: This affects the multipart message
            socket.connect(f'tcp://{address}')
            return socket
        except zmq.ZMQError as e:
            print(f"Failed to create ZMQ socket: {e}")
            raise
    
    def receiveFeed(self, socket):
        while not self._running.is_set():
            try:
                msg = socket.recv_multipart()
                frame_data = msg[0]  # The binary data of the video frame
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                # Utilize ArucoDetector
                frame = self.arucoDetector.detect_and_draw_markers(frame)
                self._frame_queue.put(frame)  # Put the frame in the queue
            except zmq.ZMQError as e:
                print(f"ZMQ Error: {e}")
    
    def get_latest_frame(self):
        try:
            return self._frame_queue.get_nowait()  # Get the latest frame without blocking
        except queue.Empty:
            return None
    
    def stop(self):
        self._running.set()
        self._thread.join()

