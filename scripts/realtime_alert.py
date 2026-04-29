"""
Script: realtime_alert.py
Chức năng: Nhận luồng mạng thời gian thực, cảnh báo nếu phát hiện tấn công
Thành viên: Duy Khải
"""
import joblib
import numpy as np
import pandas as pd
import time
import random

SELECTED_FEATURES = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
    'Flow Byts/s', 'Flow Pkts/s', 'Pkt Len Mean', 'Pkt Len Std',
    'SYN Flag Cnt', 'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'URG Flag Cnt'
]

LABEL_MAP = {
    0: 'BENIGN',
    1: 'DoS',
    2: 'DDoS',
    3: 'Web Attack',
    4: 'PortScan',
    5: 'Infiltration',
    # ... cập nhật theo LabelEncoder thực tế
}

def simulate_network_flow():
    # Sinh ngẫu nhiên 1 luồng mạng hợp lệ (demo)
    return np.random.rand(len(SELECTED_FEATURES))

def main():
    model = joblib.load("../models/random_forest.pkl")
    for _ in range(10):  # Demo 10 luồng
        flow = simulate_network_flow().reshape(1, -1)
        pred = model.predict(flow)[0]
        label = LABEL_MAP.get(pred, str(pred))
        if label != 'BENIGN':
            msg = f"[ALERT] Phát hiện lưu lượng nghi ngờ: {label}."
            print(msg)
            with open("../alerts.log", "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        else:
            print("Lưu lượng bình thường.")
        time.sleep(1)

if __name__ == "__main__":
    main()
