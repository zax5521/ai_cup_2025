from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

def main():
    # 創建模型儲存資料夾
    models_dir = './saved_models'
    os.makedirs(models_dir, exist_ok=True)
    
    # 讀取訓練資訊，根據 player_id 將資料分成 80% 訓練、20% 測試
    info = pd.read_csv('./39_Training_Dataset/train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
    
    # 獲取所有可能的mode值，用於one-hot編碼
    unique_modes = sorted(info['mode'].unique()) if 'mode' in info.columns else []
    print(f"發現的模式: {unique_modes}")
    
    # 讀取特徵 CSV 檔（位於 "./tabular_data_train"）
    datapath = './tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
    
    # 根據 test_players 分組資料
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)
    
    # 儲存每個unique_id對應的資料
    unique_id_to_data = {}
    unique_id_list = []
    
    print("正在載入數據...")
    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        
        # 添加mode的one-hot編碼到特徵中
        if 'mode' in row and len(unique_modes) > 0:
            current_mode = row['mode'].iloc[0]
            # 為每個可能的mode創建one-hot編碼
            for mode in unique_modes:
                data[f'mode_{mode}'] = 1 if mode == current_mode else 0
        
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        
        # 儲存unique_id和對應的資料
        unique_id_to_data[unique_id] = {
            'data': data,
            'target': target,
            'player_id': player_id
        }
        unique_id_list.append(unique_id)
        
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)

    print(f"數據載入完成! 訓練集大小: {x_train.shape}, 測試集大小: {x_test.shape}")
    
    # 根據熱力圖分析結果定義目標變數的重要特徵，並加入mode特徵
    mode_features = [f'mode_{mode}' for mode in unique_modes] if unique_modes else []
    
    important_features = {
        'gender': ['az_mean', 'gx_mean', 'ay_mean', 'a_skewn', 'g_skewn', 'az_var'] + mode_features,
        'hold racket handed': ['gy_mean', 'g_mean', 'a_min', 'g_min', 'g_skewn', 'g_mean'] + mode_features,
        'play years': ['gx_mean', 'az_mean', 'g_max', 'a_skewn', 'g_skewn', 'a_kurt'] + mode_features,
        'level': ['gx_mean', 'gz_rms', 'ax_mean', 'g_max', 'gz_var', 'gx_var'] + mode_features
    }
    
    # 確保所有特徵都存在於數據集中
    all_columns = list(x_train.columns)
    feature_info = {}
    for target, features in important_features.items():
        valid_features = [f for f in features if f in all_columns]
        if len(valid_features) < len(features):
            missing = set(features) - set(valid_features)
            print(f"警告：為 {target} 選擇的特徵中，這些特徵不存在：{missing}")
        feature_info[target] = valid_features
        print(f"{target} 使用特徵: {valid_features}")
    
    # 標準化特徵
    scaler = MinMaxScaler()
    le_gender = LabelEncoder()
    le_hold = LabelEncoder()
    le_years = LabelEncoder()
    le_level = LabelEncoder()
    
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    
    # 取得標籤編碼
    y_train_le_gender = le_gender.fit_transform(y_train['gender'])
    y_train_le_hold = le_hold.fit_transform(y_train['hold racket handed'])
    y_train_le_years = le_years.fit_transform(y_train['play years'])
    y_train_le_level = le_level.fit_transform(y_train['level'])
    
    # 轉換測試集標籤
    y_test_le_gender = le_gender.transform(y_test['gender'])
    y_test_le_hold = le_hold.transform(y_test['hold racket handed'])
    y_test_le_years = le_years.transform(y_test['play years'])
    y_test_le_level = le_level.transform(y_test['level'])
    
    # 將標準化數據轉換為DataFrame以便特徵選擇
    X_train_df = pd.DataFrame(X_train_scaled, columns=x_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=x_test.columns)
    
    # 第一階段：按照特徵選擇提取數據子集（用於性別和持拍手）
    X_train_gender = X_train_df[feature_info['gender']].values
    X_train_hold = X_train_df[feature_info['hold racket handed']].values
    
    X_test_gender = X_test_df[feature_info['gender']].values
    X_test_hold = X_test_df[feature_info['hold racket handed']].values
    
    # 第一階段：訓練性別和持拍手模型
    print("=== 第一階段：訓練基礎模型 ===")
    clf_gender = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf_hold = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    
    print("開始訓練性別模型...")
    clf_gender.fit(X_train_gender, y_train_le_gender)
    print("性別模型訓練完成")
    
    print("開始訓練持拍手模型...")
    clf_hold.fit(X_train_hold, y_train_le_hold) 
    print("持拍手模型訓練完成")
    
    # 第二階段：獲取性別和持拍手的預測概率，用於增強球齡和等級特徵
    print("=== 第二階段：獲取基礎預測用於增強特徵 ===")
    
    # 獲取訓練集和測試集的性別和持拍手預測概率
    gender_proba_train = clf_gender.predict_proba(X_train_gender)
    hold_proba_train = clf_hold.predict_proba(X_train_hold)
    
    gender_proba_test = clf_gender.predict_proba(X_test_gender)
    hold_proba_test = clf_hold.predict_proba(X_test_hold)
    
    # 為球齡和等級模型準備增強特徵
    # 原始特徵 + 性別預測概率 + 持拍手預測概率
    X_train_years_base = X_train_df[feature_info['play years']].values
    X_train_level_base = X_train_df[feature_info['level']].values
    
    X_test_years_base = X_test_df[feature_info['play years']].values
    X_test_level_base = X_test_df[feature_info['level']].values
    
    # 增強特徵：原始特徵 + 性別和持拍手預測概率
    X_train_years = np.column_stack([X_train_years_base, gender_proba_train, hold_proba_train])
    X_train_level = np.column_stack([X_train_level_base, gender_proba_train, hold_proba_train])
    
    X_test_years = np.column_stack([X_test_years_base, gender_proba_test, hold_proba_test])
    X_test_level = np.column_stack([X_test_level_base, gender_proba_test, hold_proba_test])
    
    print(f"增強後的球齡特徵大小: {X_train_years.shape}")
    print(f"增強後的等級特徵大小: {X_train_level.shape}")
    
    # 第三階段：使用增強特徵訓練球齡和等級模型
    print("=== 第三階段：訓練增強模型 ===")
    clf_years = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf_level = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    
    print("開始訓練球齡模型（使用增強特徵）...")
    clf_years.fit(X_train_years, y_train_le_years)
    print("球齡模型訓練完成")
    
    print("開始訓練等級模型（使用增強特徵）...")
    clf_level.fit(X_train_level, y_train_le_level)
    print("等級模型訓練完成")
    
    # 保存模型
    print("保存模型...")
    joblib.dump(clf_gender, f'{models_dir}/clf_gender.joblib')
    joblib.dump(clf_hold, f'{models_dir}/clf_hold.joblib')
    joblib.dump(clf_years, f'{models_dir}/clf_years.joblib')
    joblib.dump(clf_level, f'{models_dir}/clf_level.joblib')
    
    # 保存標準化器和標籤編碼器
    joblib.dump(scaler, f'{models_dir}/scaler.joblib')
    joblib.dump(le_gender, f'{models_dir}/le_gender.joblib')
    joblib.dump(le_hold, f'{models_dir}/le_hold.joblib')
    joblib.dump(le_years, f'{models_dir}/le_years.joblib')
    joblib.dump(le_level, f'{models_dir}/le_level.joblib')
    
    # 保存特徵選擇信息和其他資訊
    joblib.dump(feature_info, f'{models_dir}/feature_info.joblib')
    joblib.dump(unique_modes, f'{models_dir}/unique_modes.joblib')
    joblib.dump(True, f'{models_dir}/hierarchical_prediction.joblib')  # 標記使用階層預測
    
    print("模型訓練完成並已儲存")
    
    # 進行預測
    predict_and_save_results(clf_gender, clf_hold, clf_years, clf_level, 
                            scaler, le_gender, le_hold, le_years, le_level,
                            unique_id_to_data, unique_id_list, feature_info, unique_modes)
    
    # 評估模型 - 使用已訓練好的模型進行評估
    evaluate_models(clf_gender, clf_hold, clf_years, clf_level,
                    X_test_gender, X_test_hold, X_test_years, X_test_level,
                    y_test_le_gender, y_test_le_hold, 
                    y_test_le_years, y_test_le_level)


def predict_and_save_results(clf_gender, clf_hold, clf_years, clf_level, 
                            scaler, le_gender, le_hold, le_years, le_level,
                            unique_id_to_data, unique_id_list, feature_info, unique_modes):
    """對每個unique_id進行預測並保存結果"""
    # 創建結果DataFrame
    results = []
    
    # 獲取play years和level的類別數量
    play_years_classes = len(le_years.classes_)
    level_classes = len(le_level.classes_)
    
    # 針對每個unique_id進行預測
    for unique_id in unique_id_list:
        data_info = unique_id_to_data[unique_id]
        data = data_info['data']  # 這個data已經包含了mode的one-hot編碼
        
        # 標準化數據
        data_scaled = pd.DataFrame(scaler.transform(data), columns=data.columns)
        
        # 第一步：使用包含mode特徵的特徵子集預測性別和持拍手
        data_gender = data_scaled[feature_info['gender']].values  # 現在包含mode特徵
        data_hold = data_scaled[feature_info['hold racket handed']].values  # 現在包含mode特徵
        
        # 獲取性別和持拍手預測概率
        gender_proba = clf_gender.predict_proba(data_gender)
        hold_proba = clf_hold.predict_proba(data_hold)
        
        # 第二步：使用階層預測 - 將性別和持拍手預測結果作為特徵
        data_years_base = data_scaled[feature_info['play years']].values  # 現在包含mode特徵
        data_level_base = data_scaled[feature_info['level']].values  # 現在包含mode特徵
        
        # 增強特徵：原始特徵(包含mode) + 性別和持拍手預測概率
        data_years_enhanced = np.column_stack([data_years_base, gender_proba, hold_proba])
        data_level_enhanced = np.column_stack([data_level_base, gender_proba, hold_proba])
        
        # 使用增強特徵預測球齡和等級
        years_proba = clf_years.predict_proba(data_years_enhanced)
        level_proba = clf_level.predict_proba(data_level_enhanced)
        
        # 計算每個組別的平均概率
        male_prob = np.mean(gender_proba[:, 1]) if len(gender_proba[0]) > 1 else np.mean(gender_proba)
        right_handed_prob = np.mean(hold_proba[:, 1]) if len(hold_proba[0]) > 1 else np.mean(hold_proba)
        
        # 處理球齡預測
        play_years_probs = []
        for i in range(play_years_classes):
            play_years_probs.append(np.mean(years_proba[:, i]))
        
        # 處理等級預測
        level_probs = []
        for i in range(level_classes):
            level_probs.append(np.mean(level_proba[:, i]))
        
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
    
    # 創建結果DataFrame
    results_df = pd.DataFrame(results)
    
    # 將數值欄位格式化為小數點後兩位
    for col in results_df.columns:
        if col != 'unique_id':  # 不處理unique_id欄位
            results_df[col] = results_df[col].round(2)
    
    # 保存結果到CSV檔
    results_df.to_csv('prediction_results_hierarchical_with_mode.csv', index=False)
    print("預測結果已保存至 prediction_results_hierarchical_with_mode.csv")


def evaluate_models(clf_gender, clf_hold, clf_years, clf_level,
                    X_test_gender, X_test_hold, X_test_years, X_test_level,
                    y_test_le_gender, y_test_le_hold, 
                    y_test_le_years, y_test_le_level):
    """使用已訓練好的模型評估性能"""
    group_size = 27
    
    # 評估性別模型
    auc_gender = evaluate_binary_model(clf_gender, X_test_gender, y_test_le_gender, group_size)
    print(f"性別 AUC: {auc_gender:.4f}")
    
    # 評估持拍手模型
    auc_hold = evaluate_binary_model(clf_hold, X_test_hold, y_test_le_hold, group_size)
    print(f"持拍手 AUC: {auc_hold:.4f}")
    
    # 評估球齡模型（使用增強特徵）
    auc_years = evaluate_multiary_model(clf_years, X_test_years, y_test_le_years, group_size)
    print(f"球齡 AUC (階層預測+mode): {auc_years:.4f}")
    
    # 評估等級模型（使用增強特徵）
    auc_level = evaluate_multiary_model(clf_level, X_test_level, y_test_le_level, group_size)
    print(f"等級 AUC (階層預測+mode): {auc_level:.4f}")
    
    # 計算最終分數
    final_score = 0.25 * (auc_gender + auc_hold + auc_years + auc_level)
    print(f"最終分數: {final_score:.4f}")
    
    return final_score


def evaluate_binary_model(model, X_test, y_test, group_size):
    """評估二分類模型性能"""
    predicted = model.predict_proba(X_test)
    # 取出正類（index 0）的概率
    predicted = [predicted[i][0] for i in range(len(predicted))]
    
    num_groups = len(predicted) // group_size 
    if sum(predicted[:group_size]) / group_size > 0.5:
        y_pred = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
    else:
        y_pred = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
    
    y_pred = [1 - x for x in y_pred]
    y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
    
    auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
    return auc_score


def evaluate_multiary_model(model, X_test, y_test, group_size):
    """評估多分類模型性能"""
    predicted = model.predict_proba(X_test)
    num_groups = len(predicted) // group_size
    y_pred = []
    for i in range(num_groups):
        group_pred = predicted[i*group_size: (i+1)*group_size]
        num_classes = len(np.unique(y_test))
        # 對每個類別計算該組內的總機率
        class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
        chosen_class = np.argmax(class_sums)
        candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
        best_instance = np.argmax(candidate_probs)
        y_pred.append(group_pred[best_instance])
    
    y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
    auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
    return auc_score


if __name__ == '__main__':
    main()