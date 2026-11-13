# DDP 訓練曲線與 Checkpoint 一致性驗證

本次檢查雙塔召回與神經排序器的分散式訓練產出，修正訓練曲線與 checkpoint 內部紀錄未逐筆核對的問題，新增訓練損失及驗證損失的修改測試。

## 運算裝置與驗證範圍

分散式訓練入口支援 CPU／Gloo 與 CUDA／NCCL，CUDA 分支依 `local_rank` 綁定 GPU，單程序與雙程序比較要求兩個可用的 CUDA GPU。

本機裝置查詢得到 CUDA GPU 數量為 0，Apple MPS 可用，現有分散式訓練證據採用 CPU／Gloo，一程序與二程序的結果保存於 `reports/ddp_cpu_reference` 與 `reports/ddp_cpu_validation`，兩份報告均標記 `gpu_result: false`。

## 重現的問題

將二程序報告 `training_history.csv` 的一筆 `train_loss` 增加 0.25，再更新子目錄及根目錄 manifest，原有驗證仍會接受該報告，因為完整步驟與最低驗證損失檢查保留通過條件，曲線內容卻已偏離 checkpoint 內保存的訓練紀錄。

## 本次修正

`src/ddp_artifacts.py` 在載入 checkpoint 後，將 CSV 與 `system.history` 依模型階段及 epoch 排序，逐筆核對欄位、座標、訓練損失與驗證損失，數值採用絕對誤差 `1e-12`、相對誤差 0。

核對失敗時，驗證會回傳 `Training history differs from the checkpoint payload`，訓練曲線核對與既有的雜湊、完整步驟、選模、推薦及評估檢查共同執行。

新增兩項測試分別修改 `train_loss` 與 `validation_loss`，兩項測試都重建檔案雜湊，確認內容核對能辨識曲線與 checkpoint 的差異。

## 實測結果

兩份既有 CPU 報告通過新版驗證，每份涵蓋 23 個檔案與 1,440 個全域訓練步驟，四份 checkpoint 及原始訓練曲線的檔案內容保持一致。

本次修改位於產出驗證流程，模型參數、推薦結果與既有訓練時間均保留，逐項驗證結果見 [本次驗證摘要](../reports/ddp_history_review.json)。

完整本機測試共 182 項通過，包含新增的兩項曲線修改測試、CPU 雙程序訓練與 MPS 檢查。

## 執行方式

安裝神經模型依賴後，以下命令會載入既有 checkpoint，核對曲線並重新計算推薦與評估結果。

```bash
python -m pip install -r requirements/requirements-neural.txt
python -m src.validate_distributed --verify-only --output reports/ddp_cpu_reference
python -m src.validate_distributed --verify-only --output reports/ddp_cpu_validation
```

完整測試包含 CPU 雙程序訓練，使用以下命令執行，本機測試同時涵蓋 MPS 且關閉 CPU fallback。

```bash
RUN_DDP_INTEGRATION=1 PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -q
```

訓練流程與前次產出檢查修正，分別見 [DDP 訓練實作](README_DDP_TRAINING.md) 與 [完整步驟及程序清理修正](README_DDP_REVIEW.md)。
