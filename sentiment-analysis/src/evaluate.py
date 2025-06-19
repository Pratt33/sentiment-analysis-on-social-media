import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model, evaluate_classification
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def evaluate_all_models(X_test, y_test, model_dir):
    results = {}
    for model_name in ['logreg', 'nb', 'svm']:
        model = load_model(os.path.join(model_dir, f'{model_name}.pkl'))
        y_pred = model.predict(X_test)
        metrics = evaluate_classification(y_test, y_pred)
        results[model_name] = metrics
        print(f"\n{model_name.upper()} Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        plot_confusion(y_test, y_pred, f'{model_name.upper()} Confusion Matrix')
    return results

if __name__ == '__main__':
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else '../output'
    from train import load_data
    import pickle
    # Load test data and vectorizer
    df = load_data('../data/training.1600000.processed.noemoticon.csv')
    X = df['clean_text']
    y = df['target']
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = load_model(os.path.join(model_dir, 'vectorizer.pkl'))
    X_test_vec = vectorizer.transform(X_test)
    evaluate_all_models(X_test_vec, y_test, model_dir) 