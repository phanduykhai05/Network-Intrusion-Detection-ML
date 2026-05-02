import pandas as pd
import os
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# 1. Định nghĩa đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Đặc trưng mục tiêu
TARGET_FEATURES = [
    'Destination Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
    'Flow Byts/s', 'Flow Pkts/s', 'Pkt Len Mean', 'Pkt Len Std',
    'SYN Flag Cnt', 'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'URG Flag Cnt'
]

def process_data():
    input_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
    print(f"📂 Đang đọc dữ liệu: {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Ép kiểu để tiết kiệm RAM
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # Mapping tên cột
    column_mapping = {
        'Total Fwd Packets': 'Tot Fwd Pkts',
        'Total Backward Packets': 'Tot Bwd Pkts',
        'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
        'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
        'Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
        'Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
        'Flow Bytes/s': 'Flow Byts/s',
        'Flow Packets/s': 'Flow Pkts/s',
        'Packet Length Mean': 'Pkt Len Mean',
        'Packet Length Std': 'Pkt Len Std',
        'SYN Flag Count': 'SYN Flag Cnt',
        'ACK Flag Count': 'ACK Flag Cnt',
        'FIN Flag Count': 'FIN Flag Cnt',
        'RST Flag Count': 'RST Flag Cnt',
        'PSH Flag Count': 'PSH Flag Cnt',
        'URG Flag Count': 'URG Flag Cnt'
    }
    df.rename(columns=column_mapping, inplace=True)
    available_features = [f for f in TARGET_FEATURES if f in df.columns]

    # Tiền xử lý Label
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    # Save label classes for real-time inference
    np.save(os.path.join(BASE_DIR, "models", "label_classes.npy"), le.classes_)

    # Scale dữ liệu
    scaler = StandardScaler()
    df[available_features] = scaler.fit_transform(df[available_features])
    joblib.dump(scaler, os.path.join(BASE_DIR, "models", "scaler.pkl"))

    X = df[available_features]
    y = df['Label']
    
    del df
    gc.collect()

    # --- CHIẾN THUẬT CÂN BẰNG MỚI (TỐI ƯU CHO RAM VÀ ĐỘ CHÍNH XÁC) ---
    print("📊 Thống kê mẫu trước khi xử lý:\n", y.value_counts())

    # Bước 1: Chỉ giảm lớp đa số (thường là nhãn có nhiều mẫu nhất)
    # Chúng ta sẽ giữ lại tối đa 100,000 mẫu của lớp đa số để máy 16GB vẫn chạy được[cite: 7]
    majority_class = y.value_counts().idxmax()
    n_samples_majority = 100000 if y.value_counts().max() > 100000 else y.value_counts().max()
    
    strategy_rus = {majority_class: n_samples_majority}
    
    print(f"📉 Đang giảm lớp đa số ({majority_class}) xuống {n_samples_majority} mẫu...")
    rus = RandomUnderSampler(sampling_strategy=strategy_rus, random_state=42)
    X_temp, y_temp = rus.fit_resample(X, y)
    
    del X, y
    gc.collect()

    # Bước 2: SMOTE để nâng các lớp thiểu số lên bằng với lớp đa số hiện tại[cite: 7]
    print("⚖️ Đang SMOTE để cân bằng tất cả các lớp...")
    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X_res, y_res = smote.fit_resample(X_temp, y_temp)

    print("✅ Thống kê mẫu sau khi cân bằng:\n", pd.Series(y_res).value_counts())

    # --- CHIA TẬP TRAIN/TEST ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # --- LƯU KẾT QUẢ ---
    train_df = pd.DataFrame(X_train, columns=available_features)
    train_df['Label'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=available_features)
    test_df['Label'] = y_test

    out_train = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    out_test = os.path.join(BASE_DIR, "data", "processed", "test.csv")
    
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    
    print(f"🎉 HOÀN THÀNH! Tập train hiện có {len(train_df)} dòng.")

if __name__ == "__main__":
    process_data("./data/processed/cleaned_data.csv", "../data/")
