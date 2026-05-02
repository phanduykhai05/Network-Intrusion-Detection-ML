import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Xác định đường dẫn gốc dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_and_evaluate():
    # 1. Load dữ liệu từ thư mục processed
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    test_path = os.path.join(BASE_DIR, "data", "processed", "test.csv")
    
    if not os.path.exists(train_path):
        print(f"❌ Không tìm thấy file: {train_path}. Hãy chạy imbalance_feature_selection.py trước!")
        return

    print("📂 Đang tải dữ liệu (có thể mất vài chục giây với file lớn)...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    X_train = train.drop('Label', axis=1)
    y_train = train['Label']
    X_test = test.drop('Label', axis=1)
    y_test = test['Label']

    # 2. Định nghĩa mô hình (Đã tối ưu siêu tốc cho máy 12 nhân)
    models = {
        'Logistic Regression (SGD)': SGDClassifier(
            loss='log_loss',   # Chính là Logistic Regression
            n_jobs=-1,         # Tận dụng đa nhân cho việc tính gradient
            learning_rate='optimal',
            max_iter=1000,
            random_state=42
        ),
        'SVM (SGD)': SGDClassifier(
            loss='hinge',      # Chính là Linear SVM
            n_jobs=-1,         # Tận dụng đa nhân
            learning_rate='optimal',
            max_iter=1000,
            random_state=42
        ),
        'Naive Bayes': GaussianNB() # NB vốn dĩ đã rất nhanh nên không cần đa nhân
    }

    # 3. Tạo thư mục models nếu chưa có
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        print(f"\n🚀 Đang huấn luyện {name} (Vui lòng đợi)...")
        model.fit(X_train, y_train)
        
        print(f"🔍 Đang dự đoán trên tập test...")
        y_pred = model.predict(X_test)
        
        print(f"=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        
        # 4. Lưu mô hình vào thư mục chuẩn
        # Đổi tên file lưu cho phù hợp
        file_name = name.replace(' ', '_').lower().replace('(', '').replace(')', '')
        save_path = os.path.join(model_dir, f"{file_name}.pkl")
        joblib.dump(model, save_path)
        print(f"💾 Đã lưu tại: {save_path}")

if __name__ == "__main__":
    train_and_evaluate()