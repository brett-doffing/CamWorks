import mediapipe as mp
from ..Plugin import Plugin

class FaceDetector(Plugin):
    """
    The FaceDetector class represents a MediaPipe-based face detector plugin.

    This class initializes and runs the face detection pipeline, drawing detected facial landmarks on the input image.

    Attributes:
        min_detection_confidence (float): Minimum detection confidence for a face to be detected.
            Default value is 0.5.
        min_tracking_confidence (float): Minimum tracking confidence for facial landmarks to be drawn.
            Default value is 0.5.
        mp_face_mesh (mediapipe.solutions.face_mesh.FaceMesh): The MediaPipe Face Mesh solution instance.
        face_mesh (mediapipe.solutions.face_mesh.FaceMesh): The face mesh processor instance.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the face detector.

        Args:
            min_detection_confidence (float, optional): Minimum detection confidence for a face to be detected.
                Defaults to 0.5.
            min_tracking_confidence (float, optional): Minimum tracking confidence for facial landmarks to be drawn.
                Defaults to 0.5.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_faces=1
        )

    def run(self, frame, camID):
        """
        Detects facial landmarks from an image and draws them on the image.

        Args:
            frame (numpy.ndarray): Input image to process.
            camID (str): Camera ID. Currently unused in this implementation.

        Returns:
            numpy.ndarray: The input image with the detected facial landmarks drawn, or the original image if no face is detected.
        """
        results = self.face_mesh.process(frame)
        if results.multi_face_landmarks:
            # Draw the keypoints on the image
            for face_landmarks in results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            return frame
        else:
            # No face detected, return the original image
            return frame
