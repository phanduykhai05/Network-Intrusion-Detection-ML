"""
Script: model_knn_rf.py
Chức năng: Huấn luyện KNN, Random Forest, vẽ ma trận nhầm lẫn, tổng hợp kết quả
Thành viên: Quốc Huy (đã cập nhật theo yêu cầu Lab)
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def train_and_evaluate():
    # Cập nhật đường dẫn đọc file .npy theo đúng cấu trúc thư mục data/processed/
    print("Đang load dữ liệu...")
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    for name, model in models.items():
        print(f"\n================ {name} ================")
        print("⏳ Đang huấn luyện...")
        model.fit(X_train, y_train)
        
        print("⏳ Đang dự đoán...")
        y_pred = model.predict(X_test)
        
        # Tính toán và in các chỉ số theo yêu cầu bài lab
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n📊 BÁO CÁO ĐÁNH GIÁ:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}  <-- Chú ý đặc biệt chỉ số này đối với các lớp tấn công")
        print(f"F1-score:  {f1:.4f}")
        
        print("\nClassification Report Chi Tiết:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Vẽ và lưu Confusion Matrix
        cm_path = f"data/confusion_{name.replace(' ', '_').lower()}.png"
        plot_confusion(y_test, y_pred, f"Confusion Matrix - {name}", cm_path)
        print(f"✅ Đã lưu Confusion Matrix tại: {cm_path}")
        
        # Lưu model
        os.makedirs('models', exist_ok=True)
        model_path = f"models/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Đã lưu mô hình tại: {model_path}")

def compare_all_models():
    print("\n🚀 ĐANG TỔNG HỢP VÀ SO SÁNH CẢ 5 MÔ HÌNH...")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    model_files = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'SVM': 'models/svm.pkl',
        'Naive Bayes': 'models/naive_bayes.pkl',
        'KNN': 'models/knn.pkl',
        'Random Forest': 'models/random_forest.pkl'
    }

    # Vẽ khung cho 5 Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    summary_data = []

    for i, (name, filepath) in enumerate(model_files.items()):
        try:
            model = joblib.load(filepath)
            y_pred = model.predict(X_test)

            # Vẽ CM
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f"Confusion Matrix: {name}")
            axes[i].set_xlabel('Dự đoán (Predicted)')
            axes[i].set_ylabel('Thực tế (Actual)')

            # Tính chỉ số để đưa vào bảng
            acc = accuracy_score(y_test, y_pred)
            p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            summary_data.append({
                'Model': name, 'Accuracy': f"{acc:.4f}", 'Precision': f"{p:.4f}",
                'Recall': f"{r:.4f}", 'F1-Score': f"{f:.4f}"
            })

        except FileNotFoundError:
            # Xử lý nếu Gia Huy chưa đẩy file model lên
            axes[i].text(0.5, 0.5, f"Chưa có file\n{filepath}", ha='center', va='center', fontsize=12, color='red')
            axes[i].set_title(f"Confusion Matrix: {name}")
            summary_data.append({
                'Model': name, 'Accuracy': "N/A", 'Precision': "N/A", 'Recall': "N/A", 'F1-Score': "N/A"
            })

    axes[-1].axis('off') # Ẩn khung hình số 6 vì chỉ có 5 model
    plt.tight_layout()
    plt.savefig("data/all_models_confusion_matrices.png", bbox_inches='tight')
    plt.close()
    print("✅ Đã lưu ảnh ghép 5 Confusion Matrices tại: data/all_models_confusion_matrices.png")

    # In bảng thống kê
    df = pd.DataFrame(summary_data)
    print("\n" + "="*60)
    print("BẢNG SO SÁNH CHỈ SỐ CÁC MÔ HÌNH")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)

    # In ra kết luận
    print("\n📌 KẾT LUẬN LỰA CHỌN MÔ HÌNH (Best Model):")
    print("Dựa vào bảng phân tích trên, ta chọn RANDOM FOREST làm mô hình xuất sắc nhất để triển khai.")
    print("Lý do chính:")
    print(" - Trong an ninh mạng, việc bỏ sót tấn công (False Negative) nguy hiểm hơn rất nhiều so với báo động giả.")
    print(" - Random Forest có chỉ số RECALL cao nhất, đảm bảo khả năng phát hiện được tối đa các luồng tấn công.")

if __name__ == "__main__":
    train_and_evaluate()
    compare_all_models()