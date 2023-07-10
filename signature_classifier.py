class SignatureClassifier:
    def __init__(self, threshold):
        self.threshold = threshold

    def classify(self, matches):
        if len(matches) > self.threshold:
            return True  # Les signatures correspondent
        else:
            return False  # Les signatures ne correspondent pas
