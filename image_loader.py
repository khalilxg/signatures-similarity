import cv2
import os

class ImageLoader:
    def __init__(self, cheques_path, specimens_path):
        self.cheques_path = cheques_path
        self.specimens_path = specimens_path

    def load_images(self):
        cheque_images = [cv2.imread(os.path.join(self.cheques_path, f'cheque{i+1}.jpg'), cv2.IMREAD_GRAYSCALE) for i in range(5)]
        specimen_images = [cv2.imread(os.path.join(self.specimens_path, f'specimen{i+1}.jpg'), cv2.IMREAD_GRAYSCALE) for i in range(5)]
        print(f"{len(cheque_images)} images de chèques chargées.")
        print(f"{len(specimen_images)} images de spécimens chargées.")
        return cheque_images, specimen_images
