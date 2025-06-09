# ai_cup_2025
台灣2025年ai cup競賽程式。
比賽網址：https://tbrain.trendmicro.com.tw/Competitions/Details/39

本程式特徵值採用官方提供的baseline code加上mode的one hot編碼形成，baseline code特徵為34個，one hot編碼特徵為10個，共計44個。
baseline code的特徵用熱力圖分析，找出關聯性最大的前六個做為特徵，並加上one hot編碼特徵10個，共計訓練特徵為16個。
訓練途中play years與level特徵表現不理想，但發現從熱力圖看到目標特徵gender、hold racket handed兩個特徵與play years、level很有關聯性，所以訓練程式採用階層訓練的方式預測，模型選用GradientBoostingClassifier。

---------------------------------------------------------------
以下為初始資料夾結構：
```text
project/
├── 39_Test_Dataset/
│   ├── test_data/
│   ├── sample_submission.csv
│   └── test_info.csv
├── 39_Training_Dataset/
│   ├── train_data/
│   ├── Readme.txt
│   └── train_info.csv
├── baseline_code_pro.py
├── correlations.py
├── tabular_data_pro.py
├── test_code_pro.py
└── train_data.py

以下為資料夾結構說明：
39_Training_Dataset  比賽訓練資料夾
39_Test_Dataset      比賽測試資料夾
baseline_code_pro.py 訓練程式
correlations.py      產生train data相關性矩陣(熱力圖)程式
tabular_data_pro.py  取特徵程式
test_code_pro.py     測試程式
train_data.py        資料合併程式

---------------------------------------------------------------

執行順序：tabular_data_pro.py -> train_data.py -> correlations.py -> baseline_code_pro.py -> test_code_pro.py
train_data.py與correlations.py可以不用執行，只是單純查看特徵的關聯性，進而去修改訓練程式中的特徵提取。
執行完成後的資料夾結構，其中prediction_results_test_hierarchical_mode.csv為預測好的檔案，並可以直接上傳至ai cup競賽。


project/
├── 39_Test_Dataset/
│   ├── test_data/
│   ├── sample_submission.csv
│   └── test_info.csv
├── 39_Training_Dataset/
│   ├── train_data/
│   ├── Readme.txt
│   └── train_info.csv
├── correlation_analysis/
├── saved_models/
├── tabular_data_test/
├── tabular_data_train/
├── baseline_code_pro.py
├── correlations.py
├── prediction_results_hierarchical_with_mode.csv
├── prediction_results_test_hierarchical_mode.csv
├── tabular_data_pro.py
├── test_code_pro.py
├── train_data.py
├── train_x.csv
└── train_y.csv
```
