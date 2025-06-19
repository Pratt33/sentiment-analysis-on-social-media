import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from preprocessing import clean_text
from utils import save_model

def load_data(path):
    df = pd.read_csv(path, encoding='latin-1', header=None)
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = df[df['target'].isin([0, 4])]
    df['target'] = df['target'].replace(4, 1)
    df['clean_text'] = df['text'].apply(clean_text)
    return df[['clean_text', 'target']]

def train_and_save_models(data_path, output_dir):
    df = load_data(data_path)
    X = df['clean_text']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    # Logistic Regression
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train_vec, y_train)
    save_model(logreg, f'{output_dir}/logreg.pkl')
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    save_model(nb, f'{output_dir}/nb.pkl')
    # SVM
    svm = LinearSVC()
    svm.fit(X_train_vec, y_train)
    save_model(svm, f'{output_dir}/svm.pkl')
    # Save vectorizer
    save_model(vectorizer, f'{output_dir}/vectorizer.pkl')
    return X_test_vec, y_test

if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training.1600000.processed.noemoticon.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output'
    train_and_save_models(data_path, output_dir) 