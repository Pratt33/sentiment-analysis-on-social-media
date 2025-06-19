import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Load data
print('Loading data...')
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = df[df['target'].isin([0, 4])]
df['target'] = df['target'].replace(4, 1)
df['sentiment'] = df['target'].map({0: 'Negative', 1: 'Positive'})

# Label Distribution
print('Generating label distribution plot...')
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df)
plt.title('Label Distribution')
plt.savefig('output/label_distribution.png', bbox_inches='tight')
plt.close()

# Wordclouds
print('Generating wordclouds...')
for label, fname in zip([1, 0], ['wordcloud_positive.png', 'wordcloud_negative.png']):
    text = ' '.join(df[df['target'] == label]['text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Wordcloud for {"Positive" if label==1 else "Negative"} Tweets')
    plt.savefig(f'output/{fname}', bbox_inches='tight')
    plt.close()

# Confusion Matrix and ROC Curve
print('Generating confusion matrix and ROC curve...')
vectorizer = joblib.load('output/vectorizer.pkl')
model = joblib.load('output/best_model.pkl')
X_vec = vectorizer.transform(df['text'])
y_true = df['target']
y_pred = model.predict(X_vec)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('output/confusion_matrix.png', bbox_inches='tight')
plt.close()

# ROC Curve
if hasattr(model, 'predict_proba'):
    y_proba = model.predict_proba(X_vec)[:,1]
else:
    y_proba = model.decision_function(X_vec)
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('output/roc_curve.png', bbox_inches='tight')
plt.close()

print('All visuals generated and saved in output/.') 