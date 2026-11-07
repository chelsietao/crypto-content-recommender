# PyTorch 雙塔召回：推論流程與產出驗證優化

本次檢查涵蓋曝光標籤、請求時間特徵、訓練與驗證區段、候選排除、推薦計算、checkpoint 載入及產出驗證，依檢查結果修正重複計算、權重狀態與驗證覆蓋問題，並重跑訓練與回歸測試。

## 1. 修正範圍

| 檢查項目 | 本次處理 | 驗證方式 |
|---|---|---|
| V7 召回計算 | 一次產生召回分數，候選挑選與推薦輸出共用這份分數 | 記錄 `retrieval_scores` 的實際呼叫次數 |
| 使用者特徵 | 已存在於 checkpoint 的使用者採用已儲存特徵，新使用者才建立初始特徵 | 記錄既有使用者的特徵重建次數 |
| 候選過濾 | 複製分數後排除已讀內容，保留呼叫端原始分數 | 核對輸入陣列，以及候選耗盡與允許已讀內容的結果 |
| checkpoint | 載入後設定評估模式，並將雙塔的 `requires_grad` 設為 `False` | 檢查模型模式與全部雙塔參數 |
| 產出清單 | 要求 manifest 包含八個指定檔案，再逐一核對 SHA-256 | 空清單、缺少檔案與額外檔案的拒絕測試 |
| 推薦資料 | 核對兩種模型、全部 checkpoint 使用者、名次、內容 ID 與分數 | 移除使用者或模型、重複資料、名次缺口及非有限分數的拒絕測試 |

## 2. 召回流程的修改

V7 的推薦入口採用 `retrieval_scores` 取得使用者向量與內容向量的內積，接著將同一份分數交給 `select_candidates`，完成已讀排除、同分排序與前 50 筆候選選取，再輸出前 10 筆推薦。

`select_candidates` 會複製輸入陣列，已讀內容的遮罩只套用在該副本，推薦輸出使用另一份候選遮罩，因此候選挑選不會改動原始分數。

使用者特徵的讀取採用明確的快取判斷，既有使用者回傳 checkpoint 中的特徵，新使用者透過空歷史建立特徵，消除字典預設值在查詢成功時仍被求值的計算。

本次以 80 位使用者逐一執行推薦，呼叫次數記錄如下。

| 模型 | 推薦次數 | 召回分數計算次數 | 使用者歷史特徵重建次數 |
|---|---:|---:|---:|
| V7 雙塔召回 | 80 | 80 | 0 |
| V8 神經排序器 | 80 | 80 | 0 |

原 V7 在每次推薦中會呼叫兩次召回計算，本次經呼叫計數確認每次執行一次，兩種模型的推薦 ID、順序與匯出分數都通過原版本結果核對。

## 3. checkpoint 狀態的修改

模型完成雙塔訓練後會凍結塔權重，這項狀態由載入流程透過 `requires_grad_(False)` 重建，雙塔與神經排序器也會設定為評估模式。

測試對儲存及重載後的模型檢查參數狀態，雙塔可訓練參數的數量為 0，推薦分數與候選 ID 通過重載核對。

## 4. 產出驗證的修改

`python -m src.run_neural --verify-only` 會先要求 manifest 覆蓋下列八個檔案，完成雜湊核對後，才讀取 checkpoint 與推薦資料。

| 檔案 | 驗證對象 |
|---|---|
| `neural_recommender.pt` | 模型權重與重建資料 |
| `model_metrics.csv` | 八模型評估指標 |
| `evaluation_details.csv` | 逐位使用者的評估明細 |
| `training_history.csv` | 每個 epoch 的訓練與驗證損失 |
| `recommendations.csv` | 兩個神經模型的推薦結果 |
| `training_examples.csv` | 曝光時間、標籤觀察窗與訓練區段 |
| `training_curves.png` | 訓練曲線圖 |
| `run_summary.json` | 設定、環境與訓練摘要 |

推薦資料需要包含兩個神經模型與 checkpoint 內的 80 位使用者，共 160 組模型與使用者配對，每組資料會核對名次連續性、內容唯一性、有限分數，以及重新推論後的內容 ID、順序與分數。

新增測試在修改推薦資料後會重算該檔案的雜湊，再執行驗證，確認缺漏與錯誤由資料完整性檢查攔截，測試涵蓋缺少使用者、缺少模型、重複名次、重複內容、名次缺口與 NaN 分數。

## 5. 重跑訓練與指標核對

本次使用原有合成資料、種子 42 與四個時間區段，雙塔與神經排序器各訓練 30 個 epoch，依驗證損失選用第 4 與第 18 個 epoch，重建 checkpoint、訓練曲線與推薦產出。

| 模型 | Hit / Recall@10 | NDCG@10 | 候選召回率 |
|---|---:|---:|---:|
| V7 雙塔召回 | 0.1333 | 0.0452 | 0.7333 |
| V8 神經排序器 | 0.2333 | 0.1013 | 0.7333 |

八種模型的完整指標表，以及 80 位使用者的兩種神經推薦清單，均與檢查前的匯出資料逐欄核對，數值與內容相同，本次優化的交付證據為重複計算的移除、checkpoint 狀態一致性與產出驗證覆蓋。

## 6. 測試與執行方式

本次完整測試通過 50 項，新增 12 個回歸案例，涵蓋計算次數、特徵快取、候選耗盡、分數不變性、重載凍結狀態、manifest 覆蓋與推薦資料完整性。

安裝神經模型依賴後，可使用以下指令執行測試、檢查已匯出的模型與重跑訓練。

```bash
python -m pip install -r requirements-neural.txt
python -m pip check
python -m pytest -q
python -m src.run_neural --verify-only
python -m src.run_neural
python -m src.run_neural --verify-only
```

## 7. 檔案索引

| 檔案 | 內容 |
|---|---|
| [src/neural.py](src/neural.py) | 單次召回計算、使用者特徵讀取、候選過濾與 checkpoint 狀態 |
| [src/run_neural.py](src/run_neural.py) | manifest 覆蓋檢查與推薦資料驗證 |
| [tests/test_neural.py](tests/test_neural.py) | 神經模型測試與本次回歸案例 |
| [檢查結果紀錄](reports/PYTORCH_REVIEW_RESULTS.json) | 核對結果、計算次數與測試數量 |
| [模型實作文件](README_PYTORCH.md) | 資料流程、模型架構、訓練曲線與完整指標 |
| [訓練摘要](artifacts/neural/run_summary.json) | 本次重跑的環境、參數與訓練選模結果 |
| [checkpoint](artifacts/neural/neural_recommender.pt) | 重跑訓練後保存的權重與資料 |
