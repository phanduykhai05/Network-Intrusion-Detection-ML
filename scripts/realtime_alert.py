"""
Script: realtime_alert.py
Function: Simulate real-time network flow classification and generate Suricata-style alerts.
Member: Duy Khai (TASK 2.6)

Usage:
    python scripts/realtime_alert.py

Prerequisites:
    - Run eda_preprocessing.py            -> data/processed/cleaned_data.csv
    - Run imbalance_feature_selection.py  -> models/scaler.pkl, models/label_classes.npy
    - Run model_knn_rf.py                 -> models/random_forest.pkl
"""
import os
import random
import time

import joblib
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "random_forest.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")
LABELS_PATH  = os.path.join(BASE_DIR, "models", "label_classes.npy")
LOG_PATH     = os.path.join(BASE_DIR, "alerts.log")

# ── 18 core features (must match training order) ───────────────────────────
SELECTED_FEATURES = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s',
    'Pkt Len Mean', 'Pkt Len Std', 'SYN Flag Cnt',
    'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'URG Flag Cnt'
]

def simulate_network_flow():
    """Generate a random raw network flow for demonstration."""
    return np.random.rand(len(SELECTED_FEATURES))

def main():
    # Load artifacts
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun model_knn_rf.py first."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}\nRun imbalance_feature_selection.py first."
        )

    model      = joblib.load(MODEL_PATH)
    scaler     = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    if os.path.exists(LABELS_PATH):
        le_classes = np.load(LABELS_PATH, allow_pickle=True)
        print(f"Loaded {len(le_classes)} label classes.")
    else:
        # Fallback: CIC-IDS2017 class names sorted alphabetically (LabelEncoder order)
        le_classes = np.array([
            'BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk',
            'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator',
            'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator',
            'Web Attack – Brute Force', 'Web Attack – Sql Injection', 'Web Attack – XSS'
        ])
        print("[WARN] label_classes.npy not found. Using default CIC-IDS2017 class names.")

    # Validate scaler matches 18-feature input
    n_features = len(SELECTED_FEATURES)
    if scaler is not None and hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != n_features:
        print(f"[WARN] Scaler expects {scaler.n_features_in_} features but we have {n_features}. "
              "Re-run imbalance_feature_selection.py to rebuild the scaler. Using raw values for demo.")
        scaler = None

    # Clear previous log
    open(LOG_PATH, 'w', encoding='utf-8').close()

    print("Starting real-time IDS simulation (15 flows)...")
    print("-" * 60)

    for i in range(15):
        raw_flow = simulate_network_flow()

        # Scale the flow if a compatible scaler is available
        if scaler is not None:
            scaled_flow = scaler.transform(raw_flow.reshape(1, -1))
        else:
            scaled_flow = raw_flow.reshape(1, -1)

        pred_idx = model.predict(scaled_flow)[0]

        # Decode label; guard against index out of range
        if pred_idx < len(le_classes):
            label = le_classes[pred_idx]
        else:
            label = f"Unknown({pred_idx})"

        dest_port = random.choice([80, 443, 22, 21, 8080, 3389, 53])

        if str(label).upper() != 'BENIGN':
            msg = (
                f"[ALERT] Suspicious traffic detected: {label}. "
                f"Destination Port: {dest_port}."
            )
            print(msg)
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
        else:
            print(f"[INFO]  Flow {i+1:02d}: BENIGN – no alert.")

        time.sleep(0.2)

    print("-" * 60)
    print(f"Simulation complete. Alerts logged to: {LOG_PATH}")

if __name__ == "__main__":
    main()
