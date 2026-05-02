"""
Script: model_knn_rf.py
Chức năng: Huấn luyện KNN, Random Forest, vẽ ma trận nhầm lẫn, tổng hợp kết quả
Thành viên: Quốc Huy
"""
import pandas as pd
import os
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_and_evaluate():
    train = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "train.csv"))
    test  = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test.csv"))
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
        confusion_path = os.path.join(BASE_DIR, "models", f"confusion_{name.replace(' ', '_').lower()}.png")
        model_path = os.path.join(BASE_DIR, "models", f"{name.replace(' ', '_').lower()}.pkl")
        plot_confusion(y_test, y_pred, f"Confusion Matrix - {name}", confusion_path)
        joblib.dump(model, model_path)

if __name__ == "__main__":
    train_and_evaluate()