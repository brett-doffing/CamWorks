import cv2
from Camera import Camera
import numpy as np

class SIFT:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

    def find_corresponding_points(self, frame1, frame2):
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Find keypoints and descriptors using SIFT
        keypoints1, descriptors1 = self.sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(gray2, None)

        # Match keypoints between frames
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Filter out bad matches using RANSAC
        good_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.7:
                good_matches.append(m[0])
                good_matches.append(m[1])  # Append both matches

        # Apply RANSAC to filter out bad matches
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Find homography matrix using RANSAC
        # homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Filter out bad matches based on the mask
        # good_matches_filtered = []
        # for m in good_matches:
        #     if mask[m.queryIdx]:
        #         good_matches_filtered.append(m[0])
        #         good_matches_filtered.append(m[1])  # Append both matches

        # src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches_filtered]).reshape(-1, 2)
        # dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches_filtered]).reshape(-1, 2)

        # Create pairs of good matches
        good_match_pairs = []
        i = 0
        while i < len(good_matches):
            if (i + 1) % 2 != 0:  # If the current match is odd, skip it for now
                i += 1
                continue
            pair = (good_matches[i - 1], good_matches[i])
            good_match_pairs.append(pair)
            i += 1
            
        output = cv2.drawMatchesKnn(frame1, keypoints1, frame2, keypoints2, good_match_pairs, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return src_pts, dst_pts, output
