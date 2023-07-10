import cv2

class Preprocessor:

 
    
    def resize_image_specimen(self, image, size=(500, 500)):
        return cv2.resize(image, size)
    
    def resize_image_cheque(self, image, size=(500, 500)):
        return cv2.resize(image, size)
    

    # def binarize(self, image, threshold=119):
    #     # Utiliser un seuil binaire inverse puisque l'écriture est généralement plus sombre que l'arrière-plan
    #     _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    #     return image
    
    def contrast(self, image):
        alpha = 2.19 # Contraste (1.0-3.0)
        beta = 0 # Luminosité (0-100)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    

    # def convert_to_gray(self, image):
    #     return cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)


    def preprocess_cheque_image(self, image):
        # image = self.convert_to_gray(image)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = self.contrast(image)
        image = cv2.resize(image, (1000, 800))
        # image = self.binarize(image)
        return image

    def preprocess_specimen_image(self, image):
        # image = self.convert_to_gray(image)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = self.contrast(image)
        image = cv2.resize(image, (1000, 681))
        # image = self.binarize(image)
        return image

    def crop_cheque_signature(self, image):
        start_x = 690
        start_y = 440
        width = 270
        height = 155
        signature = image[start_y:start_y+height, start_x:start_x+width]
        return signature

    def crop_specimen_signature(self, image):
        start_x = 546
        start_y = 400
        width = 428
        height = 170
        signature = image[start_y:start_y+height, start_x:start_x+width]
        return signature
