"""
Script: model_lr_svm_nb.py
Chức năng: Huấn luyện Logistic Regression, SVM, Naive Bayes
Thành viên: T. Gia Huy
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import joblib

def train_and_evaluate():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    X_train = train.drop('Label', axis=1)
    y_train = train['Label']
    X_test = test.drop('Label', axis=1)
    y_test = test['Label']

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))
        joblib.dump(model, f"../models/{name.replace(' ', '_').lower()}.pkl")

if __name__ == "__main__":
    train_and_evaluate()
