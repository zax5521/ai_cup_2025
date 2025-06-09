from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv


def FFT(xreal, ximag):    
    n = 2
    while(n*2 <= len(xreal)):
        n *= 2
    
    p = int(math.log(n, 2))
    
    for i in range(0, n):
        a = i
        b = 0
        for j in range(0, p):
            b = int(b*2 + a%2)
            a = a/2
        if(b > i):
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]
            
    wreal = []
    wimag = []
        
    arg = float(-2 * math.pi / n)
    treal = float(math.cos(arg))
    timag = float(math.sin(arg))
    
    wreal.append(float(1.0))
    wimag.append(float(0.0))
    
    for j in range(1, int(n/2)):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)
        
    m = 2
    while(m < n + 1):
        for k in range(0, n, m):
            for j in range(0, int(m/2), 1):
                index1 = k + j
                index2 = int(index1 + m / 2)
                t = int(n * j / m)
                treal = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
                timag = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
                ureal = xreal[index1]
                uimag = ximag[index1]
                xreal[index1] = ureal + treal
                ximag[index1] = uimag + timag
                xreal[index2] = ureal - treal
                ximag[index2] = uimag - timag
        m *= 2
        
    return n, xreal, ximag   
    
def FFT_data(input_data, swinging_times):   
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength
       
    for num in range(len(swinging_times)-1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num+1]):
            a.append(math.sqrt(math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2)))

        a_mean[num] = (sum(a) / len(a))
        g_mean[num] = (sum(a) / len(a))
    
    return a_mean, g_mean

def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer, mode_features=None):
    allsum = []
    mean = []
    var = []
    rms = []
    XYZmean_a = 0
    a = []
    g = []
    a_s1 = 0
    a_s2 = 0
    g_s1 = 0
    g_s2 = 0
    a_k1 = 0
    a_k2 = 0
    g_k1 = 0
    g_k2 = 0
    
    for i in range(len(input_data)):
        if i==0:
            allsum = input_data[i]
            a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
            continue
        
        a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
        g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
       
        allsum = [allsum[feature_index] + input_data[i][feature_index] for feature_index in range(len(input_data[i]))]
        
    mean = [allsum[feature_index] / len(input_data) for feature_index in range(len(input_data[i]))]
    
    for i in range(len(input_data)):
        if i==0:
            var = input_data[i]
            rms = input_data[i]
            continue

        var = [var[feature_index] + math.pow((input_data[i][feature_index] - mean[feature_index]), 2) for feature_index in range(len(input_data[i]))]
        rms = [rms[feature_index] + math.pow(input_data[i][feature_index], 2) for feature_index in range(len(input_data[i]))]
        
    var = [math.sqrt((var[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    rms = [math.sqrt((rms[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    
    a_max = [max(a)]
    a_min = [min(a)]
    a_mean = [sum(a) / len(a)]
    g_max = [max(g)]
    g_min = [min(g)]
    g_mean = [sum(g) / len(g)]
    
    a_var = math.sqrt(math.pow((var[0] + var[1] + var[2]), 2))
    
    for i in range(len(input_data)):
        a_s1 = a_s1 + math.pow((a[i] - a_mean[0]), 4)
        a_s2 = a_s2 + math.pow((a[i] - a_mean[0]), 2)
        g_s1 = g_s1 + math.pow((g[i] - g_mean[0]), 4)
        g_s2 = g_s2 + math.pow((g[i] - g_mean[0]), 2)
        a_k1 = a_k1 + math.pow((a[i] - a_mean[0]), 3)
        g_k1 = g_k1 + math.pow((g[i] - g_mean[0]), 3)
    
    a_s1 = a_s1 / len(input_data)
    a_s2 = a_s2 / len(input_data)
    g_s1 = g_s1 / len(input_data)
    g_s2 = g_s2 / len(input_data)
    a_k2 = math.pow(a_s2, 1.5)
    g_k2 = math.pow(g_s2, 1.5)
    a_s2 = a_s2 * a_s2
    g_s2 = g_s2 * g_s2
    
    a_kurtosis = [a_s1 / a_s2]
    g_kurtosis = [g_s1 / g_s2]
    a_skewness = [a_k1 / a_k2]
    g_skewness = [g_k1 / g_k2]
    
    a_fft_mean = 0
    g_fft_mean = 0
    cut = int(n_fft / swinging_times)
    a_psd = []
    g_psd = []
    entropy_a = []
    entropy_g = []
    e1 = []
    e3 = []
    e2 = 0
    e4 = 0
    
    for i in range(cut * swinging_now, cut * (swinging_now + 1)):
        a_fft_mean += a_fft[i]
        g_fft_mean += g_fft[i]
        a_psd.append(math.pow(a_fft[i], 2) + math.pow(a_fft_imag[i], 2))
        g_psd.append(math.pow(g_fft[i], 2) + math.pow(g_fft_imag[i], 2))
        e1.append(math.pow(a_psd[-1], 0.5))
        e3.append(math.pow(g_psd[-1], 0.5))
        
    a_fft_mean = a_fft_mean / cut
    g_fft_mean = g_fft_mean / cut
    
    a_psd_mean = sum(a_psd) / len(a_psd)
    g_psd_mean = sum(g_psd) / len(g_psd)
    
    for i in range(cut):
        e2 += math.pow(a_psd[i], 0.5)
        e4 += math.pow(g_psd[i], 0.5)
    
    for i in range(cut):
        entropy_a.append((e1[i] / e2) * math.log(e1[i] / e2))
        entropy_g.append((e3[i] / e4) * math.log(e3[i] / e4))
    
    a_entropy_mean = sum(entropy_a) / len(entropy_a)
    g_entropy_mean = sum(entropy_g) / len(entropy_g)       
        
    # 原始特徵
    output = mean + var + rms + a_max + a_mean + a_min + g_max + g_mean + g_min + [a_fft_mean] + [g_fft_mean] + [a_psd_mean] + [g_psd_mean] + a_kurtosis + g_kurtosis + a_skewness + g_skewness + [a_entropy_mean] + [g_entropy_mean]
    
    # 加入mode的one-hot編碼特徵
    if mode_features is not None:
        output.extend(mode_features)
    
    writer.writerow(output)

def data_generate():
    # 設定數據路徑和輸出路徑
    datapath = './39_Training_Dataset/train_data'
    tar_dir = 'tabular_data_train'
    info_file = './39_Training_Dataset/train_info.csv'
    
    # datapath = './39_Test_Dataset/test_data'
    # tar_dir = 'tabular_data_test'
    # info_file = './39_Test_Dataset/test_info.csv'
    
    print(datapath)
    pathlist_txt = Path(datapath).glob('**/*.txt')
    
    # 如果有info檔案，讀取mode信息
    mode_info = {}
    unique_modes = []
    if info_file and Path(info_file).exists():
        try:
            info_df = pd.read_csv(info_file)
            if 'mode' in info_df.columns:
                unique_modes = sorted(info_df['mode'].unique())
                # 建立unique_id到mode的映射
                for _, row in info_df.iterrows():
                    mode_info[row['unique_id']] = row['mode']
                print(f"發現的模式: {unique_modes}")
            else:
                print("警告：info檔案中沒有找到mode欄位")
        except Exception as e:
            print(f"讀取info檔案時出錯: {e}")
    else:
        print("警告：沒有找到info檔案，將不使用mode特徵")
    
    # 建立表頭
    headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 
                  'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 
                  'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 
                  'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 
                  'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 
                  'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']
    
    # 加入mode的one-hot編碼表頭
    if unique_modes:
        for mode in unique_modes:
            headerList.append(f'mode_{mode}')
    
    # 確保輸出目錄存在
    Path(tar_dir).mkdir(parents=True, exist_ok=True)
    
    for file in pathlist_txt:
        f = open(file)
        
        All_data = []
        
        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)
        
        f.close()
        
        # 獲取檔案的unique_id
        unique_id = int(Path(file).stem)
        
        # 獲取該檔案對應的mode值
        current_mode = mode_info.get(unique_id, None)
        
        # 建立mode的one-hot編碼
        mode_features = []
        if unique_modes:
            for mode in unique_modes:
                if current_mode is not None:
                    mode_features.append(1 if mode == current_mode else 0)
                else:
                    # 如果沒有mode信息，使用預設值（第一個模式）
                    mode_features.append(1 if mode == unique_modes[0] else 0)
        
        swing_index = np.linspace(0, len(All_data), 28, dtype = int)

        with open('./{dir}/{fname}.csv'.format(dir = tar_dir, fname = Path(file).stem), 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i==0:
                        continue
                    feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, 
                           n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer, mode_features)
            except Exception as e:
                print("出錯檔案")
                print(Path(file).stem)
                print(f"錯誤信息: {e}")
                continue

def data_generate_train():
    """專門用於處理訓練資料的函數"""
    datapath = './39_Training_Dataset/train_data'
    tar_dir = 'tabular_data_train'
    info_file = './39_Training_Dataset/train_info.csv'
    
    pathlist_txt = Path(datapath).glob('**/*.txt')
    
    # 讀取mode信息
    mode_info = {}
    unique_modes = []
    if Path(info_file).exists():
        try:
            info_df = pd.read_csv(info_file)
            if 'mode' in info_df.columns:
                unique_modes = sorted(info_df['mode'].unique())
                # 建立unique_id到mode的映射
                for _, row in info_df.iterrows():
                    mode_info[row['unique_id']] = row['mode']
                print(f"發現的模式: {unique_modes}")
            else:
                print("警告：info檔案中沒有找到mode欄位")
        except Exception as e:
            print(f"讀取info檔案時出錯: {e}")
            return
    else:
        print("錯誤：找不到訓練資料的info檔案")
        return
    
    # 建立表頭
    headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 
                  'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 
                  'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 
                  'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 
                  'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 
                  'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']
    
    # 加入mode的one-hot編碼表頭
    for mode in unique_modes:
        headerList.append(f'mode_{mode}')
    
    # 確保輸出目錄存在
    Path(tar_dir).mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    for file in pathlist_txt:
        f = open(file)
        
        All_data = []
        
        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)
        
        f.close()
        
        # 獲取檔案的unique_id
        unique_id = int(Path(file).stem)
        
        # 獲取該檔案對應的mode值
        current_mode = mode_info.get(unique_id, None)
        if current_mode is None:
            print(f"警告：檔案 {unique_id} 沒有對應的mode信息，跳過")
            continue
        
        # 建立mode的one-hot編碼
        mode_features = []
        for mode in unique_modes:
            mode_features.append(1 if mode == current_mode else 0)
        
        swing_index = np.linspace(0, len(All_data), 28, dtype = int)

        with open('./{dir}/{fname}.csv'.format(dir = tar_dir, fname = Path(file).stem), 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i==0:
                        continue
                    feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, 
                           n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer, mode_features)
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"已處理 {processed_count} 個檔案")
            except Exception as e:
                print(f"處理檔案 {Path(file).stem} 時出錯: {e}")
                error_count += 1
                continue
    
    print(f"處理完成！成功: {processed_count}, 失敗: {error_count}")

def data_generate_test():
    """專門用於處理測試資料的函數"""
    datapath = './39_Test_Dataset/test_data'
    tar_dir = 'tabular_data_test'
    # 測試資料需要與訓練資料使用相同的mode設定
    train_info_file = './39_Training_Dataset/train_info.csv'
    
    pathlist_txt = Path(datapath).glob('**/*.txt')
    
    # 從訓練資料讀取mode信息以保持一致性
    unique_modes = []
    if Path(train_info_file).exists():
        try:
            info_df = pd.read_csv(train_info_file)
            if 'mode' in info_df.columns:
                unique_modes = sorted(info_df['mode'].unique())
                print(f"使用訓練資料的模式設定: {unique_modes}")
            else:
                print("警告：訓練資料info檔案中沒有找到mode欄位")
        except Exception as e:
            print(f"讀取訓練資料info檔案時出錯: {e}")
    else:
        print("警告：找不到訓練資料的info檔案，將不使用mode特徵")
    
    # 建立表頭
    headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 
                  'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 
                  'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 
                  'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 
                  'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 
                  'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']
    
    # 加入mode的one-hot編碼表頭
    if unique_modes:
        for mode in unique_modes:
            headerList.append(f'mode_{mode}')
    
    # 確保輸出目錄存在
    Path(tar_dir).mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    for file in pathlist_txt:
        f = open(file)
        
        All_data = []
        
        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)
        
        f.close()
        
        # 建立mode的one-hot編碼（測試資料使用預設值）
        mode_features = []
        if unique_modes:
            # 使用第一個模式作為預設值
            default_mode = unique_modes[0]
            for mode in unique_modes:
                mode_features.append(1 if mode == default_mode else 0)
        
        swing_index = np.linspace(0, len(All_data), 28, dtype = int)

        with open('./{dir}/{fname}.csv'.format(dir = tar_dir, fname = Path(file).stem), 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i==0:
                        continue
                    feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, 
                           n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer, mode_features)
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"已處理 {processed_count} 個檔案")
            except Exception as e:
                print(f"處理檔案 {Path(file).stem} 時出錯: {e}")
                error_count += 1
                continue
    
    print(f"測試資料處理完成！成功: {processed_count}, 失敗: {error_count}")

if __name__ == '__main__':
    # 可以選擇要處理的資料類型
        print("處理測試資料...")
        data_generate_train()
        print("處理測試資料...")
        data_generate_test()