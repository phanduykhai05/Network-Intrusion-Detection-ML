# Hệ thống Phát hiện Xâm nhập Mạng Thời gian thực (IDS) sử dụng Machine Learning

## Thành viên nhóm

| # | Họ tên                | MSSV/Branch   | Nhiệm vụ                                                                                   |
|---|-----------------------|--------------|-------------------------------------------------------------------------------------------|
| 1 | Duy Khải (Nhóm trưởng) | main   | Quản lý Git/GitHub, Triển khai mô hình tốt nhất, Tổng hợp sản phẩm, README, Merge code    |
| 2 | Quốc Hiếu             | N23DVCN020   | Phân tích dữ liệu (EDA) & Tiền xử lý dữ liệu                                              |
| 3 | N. Gia Huy            | N23DVCN023   | Xử lý mất cân bằng & Chọn đặc trưng                                                       |
| 4 | T. Gia Huy            | N23DVCN022   | Huấn luyện mô hình (Logistic Regression, SVM, Naive Bayes)                                |
| 5 | Quốc Huy              | N23DVCN025   | Huấn luyện mô hình (KNN, Random Forest), Vẽ ma trận nhầm lẫn, So sánh mô hình            |

---

## 📑 Mục lục
1. [Giới thiệu bài lab](#1--giới-thiệu-bài-lab)
2. [Yêu cầu bài lab](#2--yêu-cầu-bài-lab)
3. [Sản phẩm bàn giao](#3--sản-phẩm-bàn-giao)
4. [Tham khảo](#4--tham-khảo)

---

## 1. 📖 Giới thiệu bài lab
Dự án này xây dựng hệ thống phát hiện xâm nhập mạng thời gian thực (NIDS) sử dụng machine learning. Sử dụng bộ dữ liệu CIC-IDS2017, nhóm triển khai và so sánh nhiều mô hình ML (phân loại nhị phân & đa lớp), cuối cùng triển khai mô hình tốt nhất (Random Forest) để phân tích lưu lượng mạng và cảnh báo thời gian thực.

### 📊 Đặc điểm bộ dữ liệu
- **Bộ dữ liệu:** CIC-IDS2017 (Canadian Institute for Cybersecurity)
- **Loại lưu lượng:** Bình thường + nhiều loại tấn công (DoS, DDoS, Web Attack, PortScan, Infiltration, ...)
- **Đặc trưng:** 79 cột (78 số + 1 nhãn phân loại)
- **Đặc trưng luồng mạng:** Thời lượng luồng, độ dài gói, cổng, cờ TCP/UDP, byte/giây, gói/giây, ...
- **Mất cân bằng lớp:** Phần lớn mẫu là Benign.

### 🎯 Mục tiêu học tập
- Hiểu phân phối dữ liệu, tương quan và mất cân bằng trong dữ liệu mạng.
- Quản lý mã nguồn và kiểm soát phiên bản với Git/GitHub.
- Tiền xử lý dữ liệu nâng cao (chọn đặc trưng, chuẩn hóa, mã hóa, xử lý giá trị thiếu).
- Triển khai, đánh giá và so sánh nhiều mô hình ML.
- Triển khai mô hình dự đoán cho giám sát an ninh mạng thời gian thực.

---

## 2. 💻 Yêu cầu bài lab

### 2.1. Quản lý mã nguồn với Git & GitHub
- Khởi tạo repository: `Network-Intrusion-Detection-ML` trên GitHub.
- Mỗi thành viên tự commit phần code của mình để đảm bảo lịch sử commit rõ ràng.
- README.md chuyên nghiệp gồm: Giới thiệu dự án, hướng dẫn cài đặt, hướng dẫn sử dụng, tóm tắt kết quả.

### 2.2. Phân tích dữ liệu (EDA) & Tiền xử lý (Quốc Hiếu)
- Tải và gộp 8 file CSV thành một DataFrame duy nhất.
- Làm sạch dữ liệu: xóa khoảng trắng tên cột, xử lý NaN/np.inf bằng median, loại bỏ cột zero-variance & dòng trùng lặp.
- Tối ưu bộ nhớ: Downcast kiểu dữ liệu (float64→float32, int64→int16/uint8).
- Vẽ biểu đồ phân phối loại tấn công & heatmap tương quan.
- Xuất dữ liệu đã xử lý cho bước tiếp theo.

### 2.3. Xử lý mất cân bằng & Chọn đặc trưng (N. Gia Huy)
- Mã hóa nhãn bằng LabelEncoder, chuẩn hóa số học bằng StandardScaler.
- Pipeline cân bằng dữ liệu: SMOTE (tăng nhóm thiểu số lên 10% nhóm đa số), RandomUnderSampler (giảm Benign).
- Lọc giữ 18 đặc trưng cốt lõi:
  - 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s', 'Pkt Len Mean', 'Pkt Len Std',
    'SYN Flag Cnt', 'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'URG Flag Cnt'
- Chia tập train/test, xuất dữ liệu cuối cùng.

### 2.4. Huấn luyện & So sánh mô hình
- **T. Gia Huy:** Huấn luyện Logistic Regression, SVM, Naive Bayes. Sinh classification_report (Accuracy, Precision, Recall, F1-score). Đặc biệt chú ý Recall các lớp tấn công.
- **Quốc Huy:** Huấn luyện KNN, Random Forest. Sinh classification_report. Vẽ ma trận nhầm lẫn cho cả 5 mô hình. So sánh và chọn mô hình tốt nhất (Random Forest).

### 2.5. Triển khai mô hình tốt nhất & Tổng hợp sản phẩm (Duy Khải)
- Lưu mô hình Random Forest tốt nhất (.pkl, dùng joblib/pickle; nếu >100MB thì dùng Git LFS hoặc Google Drive).
- Script giả lập nhận luồng mạng thời gian thực: phân loại, log cảnh báo kiểu Suricata ra console và alerts.log.
- Gom toàn bộ code vào 1 file Jupyter Notebook (.ipynb) chạy mượt từ trên xuống dưới.
- Viết README.md (file này).

---

## 3. 📦 Sản phẩm bàn giao
- **Lịch sử commit:** Liên tục, có cấu trúc, đầy đủ thành viên.
- **Mã nguồn:** .ipynb hoặc .py, chạy mượt, có chú thích rõ ràng, hiển thị đầy đủ kết quả.
- **README.md:** Kiến trúc dự án, bảng so sánh 5 mô hình, hướng dẫn cài đặt.
- **Mô hình tốt nhất (.pkl):** File mô hình Random Forest đã huấn luyện.
- **Log file (alerts.log, tùy chọn):** Kết quả cảnh báo thời gian thực.

---

## 4. 💻 Tham khảo
- [marxgoo/Network-intrusion-detection-ml](https://github.com/marxgoo/Network-intrusion-detection-ml)
- [Kaggle: Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/code)
- [Kaggle: Intrusion Detection System Example](https://www.kaggle.com/code/ujjwalks9/intrusion-detection-system)

---

## 🧑‍💻 Phân công công việc

| Thành viên     | Nhiệm vụ                                                                                                   |
|---------------|------------------------------------------------------------------------------------------------------------|
| Quốc Hiếu      | 2.2 EDA & Tiền xử lý: Gộp CSV, làm sạch, tối ưu bộ nhớ, vẽ biểu đồ, xuất dữ liệu sạch.                     |
| N. Gia Huy     | 2.3 Mất cân bằng & 2.4 Chọn đặc trưng: Mã hóa, chuẩn hóa, cân bằng, chọn 18 đặc trưng, chia dữ liệu.       |
| T. Gia Huy     | 2.5 Huấn luyện (LR, SVM, NB): Huấn luyện, đánh giá, báo cáo, chú ý Recall lớp tấn công.                    |
| Quốc Huy       | 2.5 Huấn luyện (KNN, RF), vẽ ma trận nhầm lẫn, so sánh mô hình, chọn mô hình tốt nhất.                    |
| Duy Khải       | 2.1 GitHub, 2.6 Triển khai, 3 Tổng hợp: Repo, lưu mô hình, script cảnh báo, notebook, README               |

---

## 🚀 Cài đặt & Sử dụng

### 1. Clone repository
```bash
git clone <LINK_REPO>
cd Network-Intrusion-Detection-ML
```

### 2. Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 3. Chạy notebook hoặc script
- Mở Jupyter Notebook (`.ipynb`) và chạy lần lượt từ trên xuống dưới.
- Hoặc chạy các script Python theo hướng dẫn trong notebook.

### 4. Demo cảnh báo thời gian thực
- Sau khi huấn luyện, chạy script triển khai để mô phỏng phân loại luồng mạng thời gian thực.
- Cảnh báo sẽ hiển thị trên console và lưu vào `alerts.log`.

---

## 📊 Bảng so sánh mô hình
| Mô hình               | Độ chính xác | Precision | Recall   | F1-score |
|-----------------------|--------------|-----------|----------|----------|
| Logistic Regression   |    91.38%    |   87.25%  |  91.38%  |  88.76%  |
| SVM                   |    91.46%    |   87.37%  |  91.46%  |  88.87%  |
| Naive Bayes           |    13.77%    |   94.01%  |  13.77%  |  18.19%  |
| **KNN**               |  **97.23%**  | **98.72%**|**97.23%**|**97.88%**|
| Random Forest         |    93.17%    |   99.34%  |  93.17%  |  96.03%  |

*Đánh giá trên tập test (50,000 mẫu, weighted average). KNN đạt F1-score cao nhất trong bộ mô hình này.*

---

## 📂 Cấu trúc dự án
```
Network-Intrusion-Detection-ML/
├── data/                # Dữ liệu thô và đã xử lý
├── models/              # File mô hình đã lưu (.pkl)
├── notebooks/           # Jupyter Notebooks
├── scripts/             # Script Python
├── alerts.log           # Log cảnh báo thời gian thực
├── requirements.txt     # Thư viện Python
├── README.md            # Tài liệu dự án
└── ...
```
