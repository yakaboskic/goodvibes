import numpy as np
from tensorflow.keras.models import load_model

class DetectionModel:
    def __init__(self, models_path):
        self.amodel = load_model(f'{models_path}/acoustic_detector.keras')
        self.smodel = load_model(f'{models_path}/seismic_detector.keras')

    def predict(self, Xa, Xs):
        ya = self.amodel.predict(Xa, verbose=False)
        ys = self.smodel.predict(Xs, verbose=False)
        aavg = np.mean(ya)
        savg = np.mean(ys)
        return aavg > 0.5 or savg > 0.5
    
class ClassifierModel:
    def __init__(self, models_path):
        self.amodel = load_model(f'{models_path}/acoustic_classifier.keras')
        self.smodel = load_model(f'{models_path}/seismic_classifier.keras')
        self.labels = np.load(f'{models_path}/labels.npy')

    def predict(self, Xa, Xs):
        ya = self.amodel.predict(Xa, verbose=False)
        ys = self.smodel.predict(Xs, verbose=False)
        # Count number of times each class is predicted
        unique, counts = np.unique(np.concatenate([ya, ys]), return_counts=True)
        most_frequent_class = unique[np.argmax(counts)]
        return most_frequent_class