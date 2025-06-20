import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_model_joblib(model, path):
    joblib.dump(model, path)

def load_model_joblib(path):
    return joblib.load(path)

def evaluate_classification(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    } 