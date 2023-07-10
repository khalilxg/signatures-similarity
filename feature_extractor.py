import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=300, scaleFactor=1.21)

    def extract_features(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors
