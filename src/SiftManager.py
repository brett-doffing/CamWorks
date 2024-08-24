import cv2
import numpy as np

class SiftManager:
    def __init__(self):
        self.sift = SIFT()
        self.cameras = {}

    def add_frame(self, camID, frame):
        self.cameras[camID] = frame

    def process_frames(self):
        features = []
        for _, frame in self.cameras.items():
            kp, des = self.get_features(frame)
            # print(type(kp), type(des))
            features.append((kp, des))

        self.self_calibrate(features)

    def self_calibrate(self, features):
        points3d = []
        points2d = []

        for i in range(len(features)):
            for j in range(i+1, len(features)):
                # Match feature points between images
                matches = self.sift.match_features(features[i], features[j])

                # Compute the fundamental matrix between the two cameras
                kp1 = features[i][0]
                kp2 = features[j][0]
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                F, _ = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)

                # Decompose the fundamental matrix to get essential matrices
                Ws = []
                for k in range(len(F)):
                    W = np.array([[0, -F[k][2], F[k][1]], [F[k][2], 0, -F[k][0]], [-F[k][1], F[k][0], 0]])
                    U, _, _ = np.linalg.svd(W)
                    Ws.append(U)

                # Select the first valid essential matrix
                E = np.dot(np.dot(Ws[0], F), Ws[0].T)

                print(f"Fundamental Matrix: {F}")
                print(f"Essential Matrix: {E}")


    def get_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detect_and_compute(gray)
        return kp, des


class SIFT:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

    def detect_and_compute(self, image):
        kp, des = self.sift.detectAndCompute(image, None)
        return kp, des

    def match_features(self, features1, features2):
        des1 = features1[1]
        des2 = features2[1]
        
        matches = self.matcher.knnMatch(np.array(des1), np.array(des2), k=2)
        good_matches = []
        threshold = 0.7
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * threshold:
                good_matches.append(m[0])
        return good_matches

    def draw_matches(self, image1, kp1, image2, kp2, matches):
        img_matches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches

