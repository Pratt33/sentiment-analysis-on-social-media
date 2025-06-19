import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model, evaluate_classification
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

try:
    import scikitplot as skplt
    HAS_SCPLOT = True
except ImportError:
    HAS_SCPLOT = False

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_roc(y_true, y_proba, title):
    if HAS_SCPLOT:
        skplt.metrics.plot_roc(y_true, y_proba)
        plt.title(title)
        plt.show()
    else:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
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
        # ROC curve
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            # SVM decision_function returns shape (n_samples,) for binary
            dec = model.decision_function(X_test)
            # Convert to probability-like for ROC
            y_proba = np.zeros((len(dec), 2))
            y_proba[:,1] = (dec - dec.min()) / (dec.max() - dec.min())
            y_proba[:,0] = 1 - y_proba[:,1]
        else:
            y_proba = None
        if y_proba is not None:
            plot_roc(y_test, y_proba, f'{model_name.upper()} ROC Curve')
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