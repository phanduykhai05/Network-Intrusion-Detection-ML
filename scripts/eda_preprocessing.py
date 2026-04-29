"""
Script: eda_preprocessing.py
Chức năng: Tiền xử lý dữ liệu & phân tích EDA cho CIC-IDS2017
Thành viên: Quốc Hiếu
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_merge_csvs(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(data_dir, f)) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    return df

def clean_data(df):
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    nunique = df.nunique()
    zero_var_cols = nunique[nunique == 1].index
    df = df.drop(columns=zero_var_cols)
    df = df.drop_duplicates()
    return df

def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def plot_attack_distribution(df, save_path=None):
    plt.figure(figsize=(10,5))
    df['Label'].value_counts().plot(kind='bar')
    plt.title('Phân phối các loại tấn công')
    plt.ylabel('Số mẫu')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_correlation_heatmap(df, save_path=None):
    plt.figure(figsize=(16,12))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Heatmap tương quan')
    if save_path:
        plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    data_dir = "../data"  # Thay đổi nếu cần
    df = load_and_merge_csvs(data_dir)
    df = clean_data(df)
    df = optimize_memory(df)
    plot_attack_distribution(df, save_path="../data/attack_distribution.png")
    plot_correlation_heatmap(df, save_path="../data/corr_heatmap.png")
    df.to_csv("../data/cleaned_data.csv", index=False)
    print("Hoàn thành tiền xử lý & EDA. Dữ liệu đã lưu tại data/cleaned_data.csv")
