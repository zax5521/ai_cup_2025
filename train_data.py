import pandas as pd
import os
import numpy as np
from pathlib import Path

def generate_training_data(csv_dir, info_csv_path, output_x_path, output_y_path, include_mode=True, mode_as_onehot=True):
    """
    生成訓練數據集，將所有CSV檔案合併為一個大的訓練集
    
    參數：
    csv_dir (str): 包含所有CSV檔案的資料夾路徑
    info_csv_path (str): 包含標籤資訊的CSV檔案路徑
    output_x_path (str): 輸出的特徵資料集路徑
    output_y_path (str): 輸出的標籤資料集路徑
    include_mode (bool): 是否將mode標籤納入特徵資料集
    mode_as_onehot (bool): 是否將mode轉換為one-hot編碼
    """
    # 讀取標籤資訊
    info_df = pd.read_csv(info_csv_path)
    
    # 初始化空的DataFrame來存儲特徵和標籤
    all_features = []
    all_labels = []
    
    # 獲取所有CSV檔案的路徑
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    print(f"開始處理CSV檔案，共有 {len(info_df)} 個標籤記錄...")
    
    # 建立unique_id到檔案名稱的映射
    file_id_map = {int(f.split('.')[0]): f for f in csv_files}
    
    # 處理計數器
    processed_count = 0
    
    # 如果使用one-hot編碼，先獲取所有可能的mode值
    if include_mode and mode_as_onehot and 'mode' in info_df.columns:
        unique_modes = info_df['mode'].unique()
        mode_columns = [f'mode_{mode}' for mode in unique_modes]
    else:
        mode_columns = []
    
    # 遍歷標籤資訊中的每一行
    for idx, row in info_df.iterrows():
        unique_id = row['unique_id']
        
        # 檢查對應的CSV檔案是否存在
        if unique_id not in file_id_map:
            print(f"警告：找不到unique_id={unique_id}的CSV檔案，跳過。")
            continue
        
        # 讀取特徵CSV檔案
        try:
            file_path = os.path.join(csv_dir, file_id_map[unique_id])
            features_df = pd.read_csv(file_path)
            
            # 獲取對應的標籤資訊
            labels = row[['gender', 'hold racket handed', 'play years', 'level']]
            
            # 如果包含mode，將其加入特徵中
            if include_mode and 'mode' in row:
                if mode_as_onehot:
                    # 創建mode的one-hot編碼
                    current_mode = row['mode']
                    for mode_col in mode_columns:
                        mode_value = int(mode_col.split('_')[1])
                        features_df[mode_col] = 1 if mode_value == current_mode else 0
                else:
                    # 直接使用數值
                    features_df['mode'] = row['mode']
            
            # 為每一行特徵創建一個對應的標籤行
            num_rows = len(features_df)
            labels_df = pd.DataFrame([labels] * num_rows)
            
            # 添加到總特徵和標籤中
            all_features.append(features_df)
            all_labels.append(labels_df)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"已處理 {processed_count} 個檔案")
                
        except Exception as e:
            print(f"處理unique_id={unique_id}的檔案時出錯：{e}")
    
    # 合併所有特徵和標籤
    train_x = pd.concat(all_features, ignore_index=True)
    train_y = pd.concat(all_labels, ignore_index=True)
    
    print(f"特徵資料集大小：{train_x.shape}")
    print(f"標籤資料集大小：{train_y.shape}")
    
    # 儲存特徵和標籤資料集
    train_x.to_csv(output_x_path, index=False)
    train_y.to_csv(output_y_path, index=False)
    
    print(f"已將特徵資料集儲存至 {output_x_path}")
    print(f"已將標籤資料集儲存至 {output_y_path}")
    
    return train_x, train_y

if __name__ == '__main__':
    # 指定路徑
    csv_dir = './tabular_data_train'  # 您存放CSV檔案的資料夾
    info_csv_path = './39_Training_Dataset/train_info.csv'  # 您的標籤資訊檔案
    output_x_path = './train_x.csv'  # 輸出的特徵資料集
    output_y_path = './train_y.csv'  # 輸出的標籤資料集
    
    # 生成訓練資料集，包含mode特徵
    train_x, train_y = generate_training_data(
        csv_dir, 
        info_csv_path, 
        output_x_path, 
        output_y_path, 
        include_mode=True,  # 包含mode
        mode_as_onehot=True  # 將mode轉換為one-hot編碼
    )