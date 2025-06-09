from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os
import sys
import argparse
import time

def load_models_and_predict(test_datapath='./tabular_data_test', models_dir='./saved_models', output_file=None, show_labels=True):
    """
    載入已儲存的模型並對新數據進行預測，支援階層預測和mode特徵
    
    您的標籤定義：
    - 男生 = 1, 女生 = 2
    - 右手 = 1, 左手 = 2
    
    參數:
    test_datapath (str): 測試數據的路徑，預設為 './tabular_data_test'
    models_dir (str): 已儲存模型的路徑，預設為 './saved_models'
    output_file (str): 輸出檔案名稱，如果為None則使用預設名稱
    show_labels (bool): 是否顯示模型標籤順序
    
    返回:
    None - 將結果儲存為 CSV 檔案
    """
    start_time = time.time()

    # 設置輸出檔案名稱
    if output_file is None:
        output_file = 'prediction_results_test_hierarchical_mode.csv'
    
    print(f"開始載入模型和進行預測...")
    print(f"測試數據路徑: {test_datapath}")
    print(f"模型路徑: {models_dir}")
    print(f"輸出檔案: {output_file}")
    print(f"標籤定義: 男生=1, 女生=2; 右手=1, 左手=2")
    
    # 檢查模型資料夾是否存在
    if not os.path.exists(models_dir):
        print(f"錯誤：找不到模型資料夾 {models_dir}。請先訓練模型。")
        return
        
    # 檢查模型檔案是否存在
    required_files = ['clf_gender.joblib', 'clf_hold.joblib', 'clf_years.joblib', 
                      'clf_level.joblib', 'scaler.joblib', 'le_gender.joblib',
                      'le_hold.joblib', 'le_years.joblib', 'le_level.joblib']
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(models_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"錯誤：找不到以下模型檔案: {', '.join(missing_files)}")
        return
    
    # 載入模型
    print("正在載入模型...")
    try:
        clf_gender = joblib.load(f'{models_dir}/clf_gender.joblib')
        clf_hold = joblib.load(f'{models_dir}/clf_hold.joblib')
        clf_years = joblib.load(f'{models_dir}/clf_years.joblib')
        clf_level = joblib.load(f'{models_dir}/clf_level.joblib')
        
        # 載入標準化器和標籤編碼器
        scaler = joblib.load(f'{models_dir}/scaler.joblib')
        le_gender = joblib.load(f'{models_dir}/le_gender.joblib')
        le_hold = joblib.load(f'{models_dir}/le_hold.joblib')
        le_years = joblib.load(f'{models_dir}/le_years.joblib')
        le_level = joblib.load(f'{models_dir}/le_level.joblib')
        
        # 載入特徵選擇資訊
        feature_info = None
        feature_info_file = f'{models_dir}/feature_info.joblib'
        if os.path.exists(feature_info_file):
            feature_info = joblib.load(feature_info_file)
            print("載入特徵選擇資訊成功")
        else:
            print("警告：沒有找到特徵選擇資訊檔案")
            return
        
        # 載入unique_modes信息
        unique_modes = []
        unique_modes_file = f'{models_dir}/unique_modes.joblib'
        if os.path.exists(unique_modes_file):
            unique_modes = joblib.load(unique_modes_file)
            print(f"載入模式信息: {unique_modes}")
        else:
            print("警告：沒有找到模式信息檔案")
        
        # 檢查是否使用階層預測
        hierarchical_file = f'{models_dir}/hierarchical_prediction.joblib'
        use_hierarchical = False
        if os.path.exists(hierarchical_file):
            use_hierarchical = joblib.load(hierarchical_file)
            print(f"使用階層預測: {use_hierarchical}")
            
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        return
    
    # 檢查模型的標籤順序並輸出以供參考
    if show_labels:
        try:
            print("\n===== 模型標籤順序 =====")
            print(f"性別類別順序: {le_gender.classes_}")
            print(f"持拍手類別順序: {le_hold.classes_}")
            print(f"球齡類別順序: {le_years.classes_}")
            print(f"等級類別順序: {le_level.classes_}")
            print("=======================\n")
        except:
            print("無法獲取模型標籤順序")
    
    # 檢查測試資料夾是否存在
    if not os.path.exists(test_datapath):
        print(f"錯誤：找不到測試資料夾 {test_datapath}。")
        return
    
    # 載入測試數據
    test_datalist = list(Path(test_datapath).glob('**/*.csv'))
    
    if not test_datalist:
        print(f"警告：在 {test_datapath} 中找不到CSV檔案。")
        return
    
    print(f"發現 {len(test_datalist)} 個測試檔案")
    
    # 儲存結果
    results = []
    
    # 獲取play years和level的類別數量
    play_years_classes = len(le_years.classes_)
    level_classes = len(le_level.classes_)
    
    # 存儲所有可能的列名，用於為出錯檔案創建零值記錄
    column_names = ['unique_id', 'gender', 'hold racket handed']
    for i in range(play_years_classes):
        column_names.append(f'play years_{i}')
    for name in le_level.classes_:
        column_names.append(f'level_{name}')
    
    success_count = 0
    error_count = 0
    
    # 針對每個測試檔案進行預測
    for file in test_datalist:
        unique_id = int(Path(file).stem)
        try:
            # 讀取數據
            data = pd.read_csv(file)
            
            # 檢查CSV檔案是否已經包含mode特徵
            mode_columns = [f'mode_{mode}' for mode in unique_modes] if unique_modes else []
            has_mode_features = all(col in data.columns for col in mode_columns)
            
            if not has_mode_features and len(unique_modes) > 0:
                # 只有當CSV檔案沒有mode特徵時才添加
                default_mode = unique_modes[0]
                print(f"警告：測試數據沒有mode信息，對檔案 {unique_id} 使用預設模式 {default_mode}")
                
                for mode in unique_modes:
                    data[f'mode_{mode}'] = 1 if mode == default_mode else 0
            elif has_mode_features:
                # CSV檔案已經包含mode特徵，不需要額外處理
                pass
            
            # 標準化數據
            data_scaled = pd.DataFrame(scaler.transform(data), columns=data.columns)
            
            # 第一步：使用包含mode特徵的特徵子集預測性別和持拍手
            if feature_info:
                data_gender = data_scaled[feature_info['gender']].values
                data_hold = data_scaled[feature_info['hold racket handed']].values
            else:
                # 如果沒有特徵選擇資訊，使用所有特徵
                data_gender = data_scaled.values
                data_hold = data_scaled.values
            
            # 獲取性別和持拍手預測概率
            gender_proba = clf_gender.predict_proba(data_gender)
            hold_proba = clf_hold.predict_proba(data_hold)
            
            if use_hierarchical and feature_info:
                # 使用階層預測：將性別和持拍手預測結果作為特徵
                data_years_base = data_scaled[feature_info['play years']].values
                data_level_base = data_scaled[feature_info['level']].values
                
                # 增強特徵：原始特徵(包含mode) + 性別和持拍手預測概率
                data_years_enhanced = np.column_stack([data_years_base, gender_proba, hold_proba])
                data_level_enhanced = np.column_stack([data_level_base, gender_proba, hold_proba])
                
                # 使用增強特徵預測球齡和等級
                years_proba = clf_years.predict_proba(data_years_enhanced)
                level_proba = clf_level.predict_proba(data_level_enhanced)
            else:
                # 原始預測方法
                if feature_info:
                    data_years = data_scaled[feature_info['play years']].values
                    data_level = data_scaled[feature_info['level']].values
                else:
                    data_years = data_scaled.values
                    data_level = data_scaled.values
                
                years_proba = clf_years.predict_proba(data_years)
                level_proba = clf_level.predict_proba(data_level)
            
            # 找出男生和右手對應的索引
            male_index = None
            for i, label in enumerate(le_gender.classes_):
                if label == 1 or label == '1' or label == 1.0:
                    male_index = i
                    break
            
            right_hand_index = None
            for i, label in enumerate(le_hold.classes_):
                if label == 1 or label == '1' or label == 1.0:
                    right_hand_index = i
                    break
            
            # 獲取正確的概率
            if male_index is not None and len(gender_proba[0]) > male_index:
                male_prob = np.mean(gender_proba[:, male_index])
            else:
                print(f"警告：無法找到男生標籤 (1) 的對應索引，使用第一列概率並反轉。")
                male_prob = 1.0 - np.mean(gender_proba[:, 0])
            
            if right_hand_index is not None and len(hold_proba[0]) > right_hand_index:
                right_handed_prob = np.mean(hold_proba[:, right_hand_index])
            else:
                print(f"警告：無法找到右手標籤 (1) 的對應索引，使用第一列概率並反轉。")
                right_handed_prob = 1.0 - np.mean(hold_proba[:, 0])
            
            # 處理球齡預測
            play_years_probs = []
            for i in range(play_years_classes):
                if i < years_proba.shape[1]:
                    play_years_probs.append(np.mean(years_proba[:, i]))
                else:
                    play_years_probs.append(0.0)
            
            # 處理等級預測
            level_probs = []
            for i in range(level_classes):
                if i < level_proba.shape[1]:
                    level_probs.append(np.mean(level_proba[:, i]))
                else:
                    level_probs.append(0.0)
            
            # 創建結果字典
            result = {'unique_id': unique_id, 'gender': male_prob, 'hold racket handed': right_handed_prob}
            
            # 添加球齡預測概率
            for i, prob in enumerate(play_years_probs):
                result[f'play years_{i}'] = prob
            
            # 添加等級預測概率  
            level_names = le_level.classes_
            for i, prob in enumerate(level_probs):
                result[f'level_{level_names[i]}'] = prob
            
            results.append(result)
            success_count += 1

            if success_count % 100 == 0:
                print(f"已處理 {success_count}/{len(test_datalist)} 個檔案 ({(success_count/len(test_datalist)*100):.1f}%)")
            
        except Exception as e:
            print(f"處理檔案 {file} 時發生錯誤: {e}")
            
            # 為出錯的檔案創建零值記錄
            zero_result = {'unique_id': unique_id}
            for col in column_names[1:]:
                zero_result[col] = 0.0
                
            results.append(zero_result)
            error_count += 1
            print(f"檔案 {unique_id} 發生錯誤，將所有預測值設為零")
    
    if not results:
        print("沒有成功的預測結果。")
        return
    
    # 創建結果DataFrame並排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('unique_id')
    
    # 將數值欄位格式化為小數點後兩位
    for col in results_df.columns:
        if col != 'unique_id':
            results_df[col] = results_df[col].round(2)
    
    # 保存結果到CSV檔
    results_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"預測完成!")
    print(f"成功處理: {success_count} 個檔案")
    print(f"失敗處理: {error_count} 個檔案")
    print(f"預測結果已保存至 {output_file}")
    print(f"總執行時間: {execution_time:.2f} 秒")
    
    # 顯示預測結果的統計信息
    if success_count > 0:
        print(f"\n=== 預測結果統計 ===")
        print(f"性別預測平均值: {results_df['gender'].mean():.3f}")
        print(f"持拍手預測平均值: {results_df['hold racket handed'].mean():.3f}")
        
        # 顯示球齡分布
        years_cols = [col for col in results_df.columns if col.startswith('play years_')]
        if years_cols:
            print("球齡預測分布:")
            for col in years_cols:
                print(f"  {col}: {results_df[col].mean():.3f}")
        
        # 顯示等級分布
        level_cols = [col for col in results_df.columns if col.startswith('level_')]
        if level_cols:
            print("等級預測分布:")
            for col in level_cols:
                print(f"  {col}: {results_df[col].mean():.3f}")


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='使用已訓練的階層預測模型進行預測')
    parser.add_argument('--test', type=str, default='./tabular_data_test',
                        help='測試數據的路徑 (預設: ./tabular_data_test)')
    parser.add_argument('--models', type=str, default='./saved_models',
                        help='模型資料夾的路徑 (預設: ./saved_models)')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出檔案名稱 (預設: prediction_results_test_hierarchical_mode.csv)')
    parser.add_argument('--no-show-labels', action='store_true',
                        help='不顯示模型標籤順序')
    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行參數
    args = parse_arguments()
    # 執行預測
    load_models_and_predict(args.test, args.models, args.output, not args.no_show_labels)