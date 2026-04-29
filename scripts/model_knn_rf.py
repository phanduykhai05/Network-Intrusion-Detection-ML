"""
Script: model_knn_rf.py
Chức năng: Huấn luyện KNN, Random Forest, vẽ ma trận nhầm lẫn, tổng hợp kết quả
Thành viên: Quốc Huy
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    X_train = train.drop('Label', axis=1)
    y_train = train['Label']
    X_test = test.drop('Label', axis=1)
    y_test = test['Label']

    models = {
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))
        plot_confusion(y_test, y_pred, f"Confusion Matrix - {name}", f"../models/confusion_{name.replace(' ', '_').lower()}.png")
        joblib.dump(model, f"../models/{name.replace(' ', '_').lower()}.pkl")

if __name__ == "__main__":
    train_and_evaluate()
