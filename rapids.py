from tkinter import N
import cudf
import cuml
#from cuml import make_regression, train_test_split
from cuml.linear_model import LinearRegression as cuLinearRegression
import cupy as cp
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
from cuml.metrics.regression import r2_score
from cuml.preprocessing.LabelEncoder import LabelEncoder as cuLabelEncoder
from cuml.model_selection import train_test_split as cu_train_test_split
from cuml.metrics import accuracy_score as cu_accuracy_score




class RAPIDS_Classifier:

    def __init__(self, random_state=42) -> None:
        
        self.random_state = random_state

        self.classifier = None
        self.label_encoder = None
        self.train_data = {}
        self.train_stats = {}
    
    def init_classifier(self,model_type, **model_params):
        if model_type == "linear_regression":
            self.classifier = cuLinearRegression(**model_params)
        elif model_type == "logistic_regression":
            self.classifier = cuLogisticRegression(**model_params)
        elif model_type == "random_forest":
            self.classifier = cuRandomForestClassifier(**model_params)
        else:
            raise ValueError("Unsupported classifier type")
        self.label_encoder = cuLabelEncoder()
        return self.classifier
    
    def process_training_data(self, X, y, validation_split:float= 0.125):
        """
        X: features
        y: label (target)
        """
               
        # Ensure float32 for GPU efficiency
        X = cp.asarray(X, dtype=cp.float32)
        y_encoded = self.label_encoder.fit_transform(y)
        
            
        X_train, X_val, y_train, y_val = cu_train_test_split(X, y_encoded, test_size=validation_split, random_state=self.random_state)

        self.train_data = {
            "X_train" : X_train,
            "X_val" : X_val,
            "y_train" : y_train,
            "y_val" : y_val
        }


    def save_status(self, out_dir):
        import os
        self.save_classifier(os.path.join(out_dir, "classifier.pkl"))
        self.save_label_encoder(os.path.join(out_dir, "encoder.pkl"))
        self.save_train_stats(os.path.join(out_dir, "training_acc.json"))

    def save_label_encoder(self, out_path):
        import pickle
        with open(out_path, "wb") as fh:
            pickle.dump(self.label_encoder, fh)

    def save_classifier(self, out_path):
        import pickle
        with open(out_path, "wb") as fh:
            pickle.dump(self.classifier, fh)
    
    def save_train_stats(self, out_path):
        import json
        with open(out_path, "w") as fh:
            json.dump(self.train_stats, fh)

    def load_classifier(self, model_path):
        import pickle
        with open(model_path, "rb") as fh:
            self.classifier = pickle.load(fh)
        return self.classifier
    
    def load_encoder(self, encoder_path):
        import pickle
        with open(encoder_path, "rb") as fh:
            self.label_encoder = pickle.load(fh)
        return self.label_encoder
    
    def get_decoded_labels(self, y_pred_encoded):
        return self.label_encoder.inverse_transform(y_pred_encoded) 
    
    def train(self, X, y):
        if self.classifier is None:
            raise ValueError("classifier has not been initiated!")
        self.classifier.fit(X, y)
        return self.classifier
    
    def predict(self, X):
        return self.classifier.predict(X)

    @staticmethod
    def get_accuracy(y_hat, y_val):
        return float(cu_accuracy_score(y_val, y_hat))
    
    def process_training(self, features_data, label_data, test_split:float=0.125):
        if self.classifier is None:
            raise ValueError("classifier has not been initiated!")

        self.process_training_data(
            features_data,
            label_data,
            test_split
        )

        self.train(self.train_data["X_train"], self.train_data["y_train"])
        train_predict = self.predict(self.train_data["X_train"])
        train_acc = self.get_accuracy(train_predict, self.train_data["y_train"])
        test_predict = self.predict(self.train_data["X_val"])
        test_acc =  self.get_accuracy(test_predict, self.train_data["y_val"])
        self.train_stats.update({
            "training_acc" : train_acc,
            "test_acc" : test_acc
        })


    

