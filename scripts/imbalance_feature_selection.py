"""
Script: imbalance_feature_selection.py
Chức năng: Xử lý mất cân bằng & chọn đặc trưng
Thành viên: N. Gia Huy
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

SELECTED_FEATURES = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
    'Flow Byts/s', 'Flow Pkts/s', 'Pkt Len Mean', 'Pkt Len Std',
    'SYN Flag Cnt', 'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'URG Flag Cnt'
]

def process_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df['Label'] = LabelEncoder().fit_transform(df['Label'])
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['float32', 'float64', 'int16', 'int32', 'int64']).columns
    num_cols = [col for col in num_cols if col != 'Label']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    X = df[SELECTED_FEATURES]
    y = df['Label']
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_final, y_final = rus.fit_resample(X_res, y_res)
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)
    train = pd.DataFrame(X_train, columns=SELECTED_FEATURES)
    train['Label'] = y_train
    test = pd.DataFrame(X_test, columns=SELECTED_FEATURES)
    test['Label'] = y_test
    train.to_csv("../data/train.csv", index=False)
    test.to_csv("../data/test.csv", index=False)
    print("Đã lưu train.csv và test.csv vào thư mục data/")

if __name__ == "__main__":
    process_data("./data/processed/cleaned_data.csv", "../data/")
