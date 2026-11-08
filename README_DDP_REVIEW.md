# DDP 分散式訓練驗證與程序清理修正

本次檢查涵蓋雙塔召回與神經排序器的 DDP 訓練、產出檔案、推薦評估與啟動器生命週期，修正檔案清單覆蓋不足、訓練步驟缺漏未被辨識及逾時後工作程序殘留三項問題。

訓練曲線與 checkpoint 內部紀錄的後續核對修正，請見 [訓練曲線一致性驗證](README_DDP_HISTORY_REVIEW.md)。

## 執行裝置

訓練入口支援 CPU／Gloo 與 CUDA／NCCL，CUDA 分支依 `local_rank` 綁定裝置，單程序與雙程序比較要求主機具備兩個可用的 CUDA GPU。

本專案保存的分散式訓練結果來自 CPU／Gloo，一程序與二程序均完成訓練、checkpoint 重載與推薦評估，報告的 `device` 為 `cpu`，`gpu_result` 為 `false`。

## 問題與修正

| 檢查項目 | 重現的問題 | 本次修正 |
|---|---|---|
| 完整檔案清單 | 根目錄 manifest 只保留 `comparison.json` 的雜湊時，舊驗證仍接受報告 | 依回合數與 world size 建立必要檔案清單，逐一核對根目錄與各訓練目錄的 SHA-256 |
| 全域訓練步驟 | 同時刪除兩個 rank 的同一個全域步驟，再更新雜湊，舊驗證仍接受報告 | 由訓練樣本數、epochs 與全域 batch size 建立完整座標，核對每個 rank 的每一筆步驟 |
| 逾時清理 | 啟動器逾時時，子工作程序可能繼續執行 | 啟動獨立程序群組，逾時或中斷時終止群組及可追蹤的子程序，等待後強制結束殘留程序 |

### 檔案與訓練紀錄

每份單回合比較報告的根目錄 manifest 涵蓋 23 個檔案，其中包含一程序與二程序的子目錄 manifest，兩份子目錄清單分別涵蓋 9 與 11 個訓練產出檔案。

驗證會拒絕缺漏或多出的產出檔案、錯誤雜湊及越出報告目錄的路徑，`launcher.log` 與作業系統產生的 `.DS_Store` 不列為訓練證據。

步驟紀錄按模型階段、epoch 與 batch index 核對，每一個 rank 都必須覆蓋完整座標，並符合 `global_batch[rank::world_size]` 的樣本數，重複步驟、小數樣本數及不合法時間值均會遭到拒絕。

訓練曲線必須包含 epoch 0 到訓練結束的兩個模型階段，所選 epoch 與驗證損失必須對應曲線最低點，通訊紀錄則須包含每個階段的十筆取樣，並符合模型參數位元組數。

### Checkpoint 與評估結果

驗證會載入每一份 checkpoint，核對訓練設定、訓練摘要、可重現資料集的時間切分及參數數量，再對 80 位使用者與一位新使用者執行雙塔及排序器推薦，核對共 162 組推薦的內容、順序與分數。

評估會使用合成資料的時間切分重建測試目標，重新計算模型指標與逐位使用者結果，因此兩份報告同時修改分數、指標或刪除評估對象，也會被重新推論與評估檢查辨識。

推薦分數核對使用絕對誤差 `1e-6` 與相對誤差 `1e-5`，評估數值核對使用絕對與相對誤差 `1e-10`，一程序與二程序的既有比較門檻保留在 `compare_runs`。

### 程序生命週期

POSIX 系統的訓練啟動器使用獨立 session，清理時對程序群組及追蹤到的子程序送出終止訊號，等待三秒後對仍存活的程序送出強制結束訊號，再等待三秒並回收啟動器。

子程序使用獨立 session 的情況也納入測試，程序樹追蹤受到權限限制時，程序群組清理仍會執行，原始逾時例外會傳回呼叫端，正常結束時保留原有 exit code。

## 驗證結果

兩份既有 CPU 訓練報告通過新版驗證，每份均覆蓋 23 個檔案與 1,440 個全域訓練步驟，單程序與雙程序各包含 720 個步驟，雙程序的每個 rank 均核對其 720 筆紀錄。

| 報告 | 訓練精度與比較方式 | 驗證結果 |
|---|---|---|
| `reports/ddp_cpu_reference` | FP64 訓練，參數與輸出數值比較 | checkpoint 內的 FP32 參數與推薦分數差異為 0，162 組推薦一致 |
| `reports/ddp_cpu_validation` | FP32 訓練，推薦排序比較 | 162 組推薦內容與順序一致，模型評估指標一致 |

本次更新報告格式與檔案清單，保留四份 checkpoint、訓練曲線、訓練時間與推薦指標，並以重新載入及推論核對既有結果。

新增 25 項測試涵蓋檔案缺漏、整步與整個 epoch 缺漏、分片樣本數、損失選模、通訊資料、裝置標記、推薦與評估修改，以及同群組和獨立 session 的工作程序清理，完整測試共 180 項通過，包含本機 MPS 檢查與 CPU 雙程序整合測試，結果另見 [驗證摘要](reports/ddp_verification_review.json)。

## 重現方式

安裝依賴後，使用既有報告執行完整驗證，命令會重新載入 checkpoint 並核對推薦與評估結果。

```bash
python -m pip install -r requirements-neural.txt
python -m src.validate_distributed --verify-only --output reports/ddp_cpu_reference
python -m src.validate_distributed --verify-only --output reports/ddp_cpu_validation
```

執行新增測試與 CPU 雙程序整合測試時，設定 `RUN_DDP_INTEGRATION=1`，整合測試採用 batch size 586，涵蓋尾批只有一筆樣本、另一個 rank 沒有樣本的情況。

```bash
RUN_DDP_INTEGRATION=1 python -m pytest -q tests/test_ddp_artifacts.py tests/test_distributed_training.py
RUN_DDP_INTEGRATION=1 python -m pytest -q
```

GitHub Actions 的 neural 工作包含新增測試及兩份 CPU 報告驗證，分散式訓練的模型流程與梯度加權說明見 [DDP 訓練實作](README_DDP_TRAINING.md)。

## 修改檔案

| 檔案 | 職責 |
|---|---|
| `src/ddp_artifacts.py` | 完整產出清單、訓練步驟與 checkpoint 評估核對 |
| `src/process_runner.py` | 訓練啟動器、程序群組與程序樹清理 |
| `src/validate_distributed.py` | 串接完整驗證與啟動器清理 |
| `tests/test_ddp_artifacts.py` | 產出修改與程序清理測試 |
| `requirements-neural.txt` | 明列程序管理所需的 psutil 版本 |
| `.github/workflows/verify.yml` | 納入新增測試與兩份 CPU 報告驗證 |
