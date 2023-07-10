import cv2
import numpy as np

class SimilarityCalculator:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def calculate_similarity(self, cheque_descriptors, specimen_descriptors):
        if cheque_descriptors is None or specimen_descriptors is None:
            return []
        
        matches = self.matcher.knnMatch(cheque_descriptors, specimen_descriptors, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        good_matches.sort(key=lambda x: x.distance)
        
        return good_matches
