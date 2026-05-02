"""
Script: eda_preprocessing.py
Chức năng: Tiền xử lý dữ liệu & phân tích EDA cho CIC-IDS2017
Thành viên: Quốc Hiếu (FINAL FIX)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ================= LOAD DATA =================
def load_and_merge_csvs(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    df_list = []
    for f in files:
        path = os.path.join(data_dir, f)
        print(f"Đang load: {f}")
        temp_df = pd.read_csv(path)
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    print("Shape sau khi merge:", df.shape)
    return df


# ================= CLEAN DATA =================
def clean_data(df):
    print("Cleaning data...")

    # Xóa khoảng trắng tên cột
    df.columns = df.columns.str.strip()

    # Đổi tên cột trùng (xảy ra khi merge nhiều CSV)
    cols = pd.Series(df.columns)
    seen = {}
    for i, c in enumerate(cols):
        if c in seen:
            seen[c] += 1
            cols[i] = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
    df.columns = cols

    # Thay inf → NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ✅ FIX hoàn toàn ChainedAssignmentError
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Xóa cột không có biến thiên
    nunique = df.nunique()
    zero_var_cols = nunique[nunique <= 1].index
    df.drop(columns=zero_var_cols, inplace=True)

    # Xóa duplicate
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Đã xóa {before - df.shape[0]} dòng trùng")

    return df


# ================= MEMORY OPTIMIZATION =================
def optimize_memory(df):
    print("Optimizing memory...")

    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory trước: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        # 🚫 Bỏ qua cột dạng object (Label)
        if col_type == 'object':
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        # Integer
        if str(col_type).startswith('int'):
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
            else:
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype(np.int16)

        # Float
        elif str(col_type).startswith('float'):
            df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory sau: {end_mem:.2f} MB (giảm {(start_mem - end_mem):.2f} MB)")

    return df


# ================= EDA =================
def plot_attack_distribution(df, save_path=None):
    plt.figure(figsize=(10,5))
    df['Label'].value_counts().plot(kind='bar')
    plt.title('Phân phối các loại tấn công')
    plt.ylabel('Số mẫu')
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path)

    plt.close()


def plot_correlation_heatmap(df, save_path=None):
    print("Plotting heatmap (sample)...")

    # ⚠️ Sample để tránh crash RAM
    sample_df = df.sample(n=5000, random_state=42)

    plt.figure(figsize=(14,10))
    corr = sample_df.corr(numeric_only=True)

    sns.heatmap(corr, cmap='coolwarm', center=0)

    plt.title('Heatmap tương quan')

    if save_path:
        plt.savefig(save_path)

    plt.close()


# ================= MAIN =================
if __name__ == "__main__":
    print("🚀 RUNNING FINAL SCRIPT...")

    # 👉 Fix path chuẩn
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")

    output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_merge_csvs(data_dir)
    df = clean_data(df)
    df = optimize_memory(df)

    plot_attack_distribution(
        df,
        save_path=os.path.join(BASE_DIR, "data", "attack_distribution.png")
    )

    plot_correlation_heatmap(
        df,
        save_path=os.path.join(BASE_DIR, "data", "corr_heatmap.png")
    )

    output_file = os.path.join(output_dir, "cleaned_data.csv")
    df.to_csv(output_file, index=False)

    print(f"✅ Hoàn thành! File lưu tại: {output_file}")