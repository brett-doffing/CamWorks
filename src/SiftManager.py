import cv2
import numpy as np

class SiftManager:
    def __init__(self):
        self.sift = SIFT()
        self.cameras = {}

    def add_frame(self, camID, frame):
        if camID in self.cameras:
            self.cameras[camID].append(frame)
        else:
            self.cameras[camID] = [frame]

    def process_frames(self):
        features = []
        camera0_length = len(list(self.cameras.values())[0])  # Get length of frames for camera 0
        for i in range(camera0_length):
            for _, frames in self.cameras.items():
                kp, des = self.get_features(frames[i])
                # print(type(kp), type(des))
                features.append((kp, des))

        self.self_calibrate(features)
        # points3d, points2d = self.self_calibrate(features)

        # camera_poses = []
        # intrinsics = []

    def self_calibrate(self, features):
        points3d = []
        points2d = []

        for i in range(len(features)):
            for j in range(i+1, len(features)):
                # Match feature points between images
                matches = self.sift.match_features(features[i], features[j]) # NOTE: doesn't match first and last

                # Compute the fundamental matrix between the two cameras
                src_pts = np.float32([features[i][0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([features[j][0][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
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

        #         # Triangulate points from the fundamental and essential matrices
        #         triangulated_points = []
        #         for k in range(len(src_pts)):
        #             point3d = cv2.triangulatePoints(E, F, src_pts[k], dst_pts[k])
        #             triangulated_points.append(point3d)

        #         # Store the 3D points and 2D projections
        #         points3d.extend(triangulated_points)
        #         points2d.extend([features[i][0][m.queryIdx].pt for m in matches])
        #         points2d.extend([features[j][0][m.trainIdx].pt for m in matches])

        # return points3d, points2d


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
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.7: # threshold
                good_matches.append(m[0])
        return good_matches

    def draw_matches(self, image1, kp1, image2, kp2, matches):
        img_matches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches


# # Perform bundle adjustment to refine the camera poses and 3D structure
# camera_poses = []
# intrinsics = []

# for i in range(len(features)):
#     # Compute the intrinsic parameters (focal length, principal point)
#     K = np.array([[1000.0, 0, features[i][0].pt[0]], [0, 1000.0, features[i][0].pt[1]], [0, 0, 1]])
#     camera_poses.append(K)

# # Define the bundle adjustment function
# def bundle_adjustment(camera_poses, points3d, points2d):
#     # Initialize the optimization variables
#     x = np.zeros((len(points3d), 6))  # Camera poses (x, y, z, roll, pitch, yaw)
#     w = np.random.rand(len(points3d), 1)  # Weights for robustification

#     # Define the cost function
#     def cost(x):
#         errors = []
#         for i in range(len(points3d)):
#             point3d = points3d[i]
#             camera_pose = x[i, :4]  # Extract the camera pose (x, y, z)
#             rotation_matrix = cv2.Rodrigues(camera_pose[:3])[0].T
#             projection = np.dot(rotation_matrix, point3d) + camera_pose[3:]
#             error = np.linalg.norm(projection - points2d[i])
#             errors.append(error * w[i][0])

#         return np.sum(errors)

#     # Perform the bundle adjustment using scipy's least_squares function
#     from scipy.optimize import least_squares
#     result = least_squares(cost, x)
#     camera_poses = result.x.reshape(-1, 6)
#     intrinsics = []
#     for i in range(len(camera_poses)):
#         K = np.array([[1000.0, 0, features[i][0].pt[0]], [0, 1000.0, features[i][0].pt[1]], [0, 0, 1]])
#         intrinsics.append(K)

# # Refine the camera poses and intrinsic parameters
# result = bundle_adjustment(camera_poses, points3d, points2d)

# # Print the refined camera poses and intrinsic parameters
# for i in range(len(result)):
#     print(f"Camera {i} pose: {result[i, :4]}")
#     print(f"Intrinsic parameters: {intrinsics[i]}")

# # Compute the projection matrices for each camera
# projection_matrices = []
# for i in range(len(features)):
#     K = intrinsics[i]
#     R = cv2.Rodrigues(result[i, :3])[0].T
#     t = result[i, 3:]
#     P = np.dot(K, np.hstack((R, t[:, None])))
#     projection_matrices.append(P)

# # Project the 3D points onto the image planes of each camera
# projected_points = []
# for i in range(len(features)):
#     P = projection_matrices[i]
#     projected_point = np.dot(P, points3d[0])
#     projected_points.append(projected_point)

# # Print the projected points for each camera
# for i in range(len(features)):
#     print(f"Projected point {i}: {projected_points[i]}")
