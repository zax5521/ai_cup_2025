import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def analyze_correlations(x_path, y_path, output_dir='.'):
    """
    分析特徵和目標變數之間的相關性，並生成熱力圖
    
    參數：
    x_path (str): 特徵資料集路徑
    y_path (str): 標籤資料集路徑
    output_dir (str): 輸出圖片的資料夾路徑
    """
    print("開始讀取資料...")
    
    # 創建輸出資料夾（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 讀取資料
    try:
        train_x = pd.read_csv(x_path)
        train_y = pd.read_csv(y_path)
        
        print(f"資料讀取完成。特徵資料集大小：{train_x.shape}，標籤資料集大小：{train_y.shape}")
    except Exception as e:
        print(f"讀取資料時出錯：{e}")
        return
    
    # 設置字體，使用不依賴於中文的標題
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 特徵間的相關性分析
    print("計算特徵間的相關性矩陣...")
    feature_corr = train_x.corr()
    
    # 設置圖形大小
    plt.figure(figsize=(20, 16))
    
    # 繪製特徵間相關性熱力圖
    mask = np.triu(np.ones_like(feature_corr, dtype=bool))  # 只顯示下三角
    sns.heatmap(feature_corr, mask=mask, annot=False, cmap='coolwarm', 
                vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Heatmap', fontsize=20)  # 使用英文標題
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300)
    plt.clf()  # 清除圖形
    
    # 2. 特徵與目標變數的相關性分析
    print("計算特徵與目標變數的相關性...")
    
    # 合併特徵和標籤 - 只取第一行作為樣本，因為每27行標籤相同
    # 避免計算重複值導致結果偏差
    unique_rows = train_y.iloc[::27].reset_index(drop=True)
    unique_features = train_x.iloc[::27].reset_index(drop=True)
    combined_df = pd.concat([unique_features, unique_rows], axis=1)
    
    # 計算所有特徵與目標變數的相關性
    target_vars = ['gender', 'hold racket handed', 'play years', 'level']
    feature_target_corr = pd.DataFrame()
    
    for target in target_vars:
        if target in combined_df.columns:
            corr_series = combined_df[train_x.columns].corrwith(combined_df[target])
            feature_target_corr[target] = corr_series
    
    # 繪製特徵與目標變數相關性熱力圖
    plt.figure(figsize=(12, 18))
    sns.heatmap(feature_target_corr, annot=True, cmap='coolwarm', vmin=-0.5, vmax=0.5,
                linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Feature-Target Correlation Heatmap', fontsize=20)  # 使用英文標題
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_target_correlation.png'), dpi=300)
    plt.clf()
    
    # 3. 找出與每個目標變數最相關的前10個特徵
    print("識別最相關的特徵...")
    
    for target in target_vars:
        if target in feature_target_corr.columns:
            # 僅使用特徵列，忽略其他目標變數
            top_corr = feature_target_corr[target].abs().sort_values(ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_corr.values, y=top_corr.index)
            plt.title(f'Top 10 Features Correlated with {target}', fontsize=16)
            plt.xlabel('Absolute Correlation Coefficient')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'top_features_{target}.png'), dpi=300)
            plt.clf()
    
    # 4. 特徵冗餘分析 - 找出高度相關的特徵對
    print("識別高度相關的特徵對...")
    
    # 獲取上三角矩陣
    upper_tri = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    
    # 找出相關性絕對值大於0.8的特徵對
    high_corr_pairs = [(col1, col2, upper_tri.loc[col1, col2]) 
                       for col1 in upper_tri.columns 
                       for col2 in upper_tri.columns 
                       if abs(upper_tri.loc[col1, col2]) > 0.8]
    
    # 按相關性絕對值排序
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # 將結果保存為CSV
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
        high_corr_df.to_csv(os.path.join(output_dir, 'high_correlation_features.csv'), index=False)
        print(f"已將高度相關的特徵對保存至 {os.path.join(output_dir, 'high_correlation_features.csv')}")
    else:
        print("沒有找到相關性絕對值大於0.8的特徵對")
    
    # 5. 保存相關性矩陣為CSV
    feature_corr.to_csv(os.path.join(output_dir, 'feature_correlation_matrix.csv'))
    feature_target_corr.to_csv(os.path.join(output_dir, 'feature_target_correlation.csv'))
    
    # 6. 分析每個目標變數之間的相關性
    print("分析目標變數之間的相關性...")
    if len(unique_rows) > 0:
        target_corr = unique_rows[target_vars].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                    linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Target Variables Correlation', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_correlation.png'), dpi=300)
        plt.clf()
    
    print("相關性分析完成！結果已保存至", output_dir)

if __name__ == '__main__':
    x_path = './train_x.csv'  # 您的特徵資料集路徑
    y_path = './train_y.csv'  # 您的標籤資料集路徑
    output_dir = './correlation_analysis'  # 輸出資料夾
    
    analyze_correlations(x_path, y_path, output_dir)