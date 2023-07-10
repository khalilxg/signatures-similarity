import cv2
import numpy as np
from image_loader import ImageLoader
from preprocessor import Preprocessor
from feature_extractor import FeatureExtractor
from similarity_calculator import SimilarityCalculator
from signature_classifier import SignatureClassifier

# Paths
cheques_path = r'C:\Workspace\PythonWorkSpace\im\Cheque images\Cheques'
specimens_path = r'C:\Workspace\PythonWorkSpace\im\signatures'

# Initialiser les classes
image_loader = ImageLoader(cheques_path, specimens_path)
preprocessor = Preprocessor()
feature_extractor = FeatureExtractor()
similarity_calculator = SimilarityCalculator()
signature_classifier = SignatureClassifier(threshold=50)


# Charger les images
cheque_images, specimen_images = image_loader.load_images()


# select les images pour la comparaison
cheque_id = 3
specimen_id = 1


# Affichage d'images avant prétraitement
cv2.imshow('Original Cheque Image', cheque_images[cheque_id])
cv2.imshow('Original Specimen Image', specimen_images[specimen_id])

# Prétraitement et recadrage des images
cheque_images = [preprocessor.preprocess_cheque_image(img) for img in cheque_images]
specimen_images = [preprocessor.preprocess_specimen_image(img) for img in specimen_images]

# Cropping 
cheque_images = [preprocessor.crop_cheque_signature(img) for img in cheque_images]
specimen_images = [preprocessor.crop_specimen_signature(img) for img in specimen_images]

# Redimensionner les images avant le recadrage
cheque_images = [preprocessor.resize_image_cheque(img) for img in cheque_images]
specimen_images = [preprocessor.resize_image_specimen(img) for img in specimen_images]

# Extraction des caractéristiques
cheque_features = [feature_extractor.extract_features(img) for img in cheque_images]
specimen_features = [feature_extractor.extract_features(img) for img in specimen_images]

# Dessiner les points d'intérêt de la première image de chèque
cheque_keypoints, cheque_descriptors = cheque_features[cheque_id]
cheque_image_with_keypoints = cv2.drawKeypoints(cheque_images[cheque_id], cheque_keypoints, None, color=(0, 255, 0))

cv2.imshow('Cheque Image with KeyPoints', cheque_image_with_keypoints)

# Dessiner les points d'intérêt de la première image de spécimen
specimen_keypoints, specimen_descriptors = specimen_features[specimen_id]
specimen_image_with_keypoints = cv2.drawKeypoints(specimen_images[specimen_id], specimen_keypoints, None, color=(0, 255, 0))
cv2.imshow('Specimen Image with KeyPoints', specimen_image_with_keypoints)

# Calculer la similarité
matches = similarity_calculator.calculate_similarity(cheque_descriptors, specimen_descriptors)


# Classer les signatures
is_match = signature_classifier.classify(matches)

# Imprimer le résultat
if is_match:
    print("La signature sur le chèque correspond au spécimen.")
else:
    print("La signature sur le chèque ne correspond pas au spécimen.")

# Afficher le nombre de correspondances et la distance du meilleur match
print(f'Nombre de correspondances pour l\'image de chèque {cheque_id} et le spécimen {specimen_id} : {len(matches)}')
print(f'Distance du meilleur match pour l\'image de chèque {cheque_id} et le spécimen {specimen_id} : {matches[0].distance if matches else "Pas de correspondances"}')


cv2.waitKey(0)
cv2.destroyAllWindows()
