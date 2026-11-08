# DDP 分散式訓練實作與 CPU 一致性驗證

本版本將 PyTorch DistributedDataParallel 接入雙塔召回與神經排序器的訓練流程，沿用同一份資料與時間區段，完成本機一程序、二程序的 CPU／Gloo 訓練、checkpoint 重載、推薦及梯度一致性驗證。

完整檔案清單、訓練步驟覆蓋、checkpoint 評估與逾時程序清理的修正，請見 [DDP 驗證優化紀錄](README_DDP_REVIEW.md)，兩份 CPU 報告已更新至包含子目錄 manifest 的格式。

## 1. 本次完成範圍

| 元件 | 完成內容 |
|---|---|
| 訓練入口 | 使用 torchrun 啟動程序，讀取 rank、local rank 與 world size |
| 模型流程 | 雙塔完成訓練與選模後凍結，接續建立候選特徵並訓練排序器 |
| 程序內模型 | 每個程序持有模型與 AdamW，DDP 同步梯度 |
| 全域 batch | 每個程序採用同一個全域 shuffle，按 rank 分片，保留不足整批的尾批 |
| 損失加權 | 依實際樣本數調整本地 loss，使梯度平均對應全域 batch 的平均損失 |
| 選模 | 訓練與驗證損失以全域總損失、樣本數計算，依驗證損失選擇權重 |
| 結果 | 保存各 rank 的步驟、checkpoint、訓練曲線資料、推薦指標與通訊取樣 |
| 驗證 | 檢查副本參數、分片樣本、checkpoint 重載與單程序／二程序的結果 |

本次實際執行裝置為 CPU，通訊後端為 Gloo，每個程序使用一個 CPU 運算執行緒，啟動器採用明確的 loopback 位址，結果資料的 `gpu_result` 為 `false`。

程式的 CUDA 分支會綁定 `local_rank` 對應的裝置，選用 NCCL，並在裝置數不足時中止，執行入口與批次腳本包含此條件檢查。

## 2. 全域 batch 與梯度

全域 batch size 設為 128，一程序與二程序採用相同亂數種子及 shuffle 次序，每個 rank 取得 `global_batch[rank::world_size]`，尾批依實際樣本數處理。

本地損失使用類別加權 BCE 的總和，再乘以 `world_size / global_batch_samples`，DDP 對各 rank 的梯度取平均後，得到全域 batch 平均損失的梯度。

某個 rank 沒有實際樣本時，程式用一筆輸入建立零損失的計算圖，讓該 rank 參與梯度同步，實際樣本數記為 0，步驟紀錄會核對各 rank 樣本數總和等於全域 batch。

梯度單元測試包含 1、3、13 筆樣本與 1、2、4 個分片，另以全域 batch 586 執行二程序整合測試，涵蓋雙塔訓練資料尾批只有一筆樣本的情況。

## 3. 資料與選模一致性

每個訓練階段會核對程序間輸入的 SHA-256，驗證損失以不同 rank 的總損失與樣本數合併計算，選定權重後再核對各程序的參數，副本間最大參數差需要為 0。

單程序與二程序的比較會核對標籤分區、排序器候選範例索引，以及模型輸出，FP64 數值參考包含參數與分數誤差門檻，FP32 排序核對要求推薦 ID、順序與評估指標一致，並保存數值差異。

兩種精度都各訓練雙塔與排序器 30 個 epoch，全域 batch 為 128，雙塔選用第 4 個 epoch，排序器選用第 18 個 epoch，兩個階段的程序間副本參數最大差均為 0。

## 4. CPU 驗證結果

以下結果使用 80 位既有使用者與 1 位新使用者，每位分別核對 V7、V8，因此每次比較包含 162 組模型與使用者配對。

| 檢查          |   雙塔儲存參數最大差 |   排序器儲存參數最大差 |   推薦分數最大差 |   推薦 ID 與順序一致組數 |
|:--------------|---------------------:|-----------------------:|-----------------:|-------------------------:|
| FP64 數值參考 |            0         |             0          |      0           |                      162 |
| FP32 排序核對 |            1.043e-07 |             0.00063546 |      0.000105023 |                      162 |

FP64 用於數值參考，訓練後轉回模型既有的 FP32 參數格式保存，本次一程序與二程序的儲存參數及推薦分數最大差為 0。

FP32 的推薦 ID、順序與評估指標通過核對，雙塔參數最大差為 0.0000001043，排序器參數最大差為 0.0006354600，推薦分數最大差為 0.0001050234，這些差異保留於比較紀錄。

## 5. 本次 CPU 計時與通訊紀錄

| CPU 訓練精度   |   程序數 |   資料準備與 fit 秒 |   雙塔訓練與選模秒 |   排序器訓練與選模秒 |
|:---------------|---------:|--------------------:|-------------------:|---------------------:|
| FP64           |        1 |              5.27   |             0.3008 |               0.055  |
| FP64           |        2 |              5.768  |             0.4668 |               0.1057 |
| FP32           |        1 |              5.4673 |             0.2927 |               0.082  |
| FP32           |        2 |              6.1824 |             0.4709 |               0.1104 |

`fit` 計時包含資料準備、特徵建立及兩個階段的訓練與選模，取各 rank 的最大值，獨立通訊測試及檔案匯出在這段計時之外執行，階段計時保存各模型訓練與選模的耗時。

通訊測試以該階段參數的相同資料型別及字節數建立向量，先暖機 5 次，再量測 10 次 all-reduce，每次等待運算完成並取所有 rank 的最長時間，結果保存於 `communication.csv`。

每一步另記錄前向、包含 DDP 同步的反向，以及 optimizer 耗時，這些紀錄來自本次 CPU／Gloo 執行。

## 6. checkpoint 與產出驗證

每次訓練由 rank 0 保存 checkpoint，重新載入後核對 81 位使用者的推薦 ID 與分數，各 rank 分別保存身分與步驟紀錄，manifest 保存產出檔案的 SHA-256。

`--verify-only` 會核對雜湊、程序身分、回合覆蓋、單程序與二程序的比較結果，以及每個全域 batch 的樣本數，若樣本重複或遺漏，驗證會以錯誤結束。

## 7. 執行方式

安裝專案依賴後，可用下列指令重現本次 CPU 數值參考與排序核對。

```bash
python -m pip install -r requirements-gpu.txt
python -m src.validate_distributed --device cpu --precision float64 \
  --epochs 30 --output reports/ddp_cpu_reference
python -m src.validate_distributed --device cpu --precision float32 \
  --comparison ranking --epochs 30 --output reports/ddp_cpu_validation
python -m src.validate_distributed --verify-only --output reports/ddp_cpu_reference
python -m src.validate_distributed --verify-only --output reports/ddp_cpu_validation
RUN_DDP_INTEGRATION=1 python -m pytest -q
```

本次完整測試包含 MPS 與本機二程序 DDP，155 項通過，GitHub Actions 會執行 CPU／Gloo 整合測試與已匯出結果的驗證。

## 8. 程式與證據索引

| 檔案 | 內容 |
|---|---|
| [src/distributed_training.py](src/distributed_training.py) | DDP 訓練、樣本加權、選模與通訊取樣 |
| [src/train_distributed.py](src/train_distributed.py) | torchrun 工作程序入口與輸出 |
| [src/validate_distributed.py](src/validate_distributed.py) | 一程序／二程序比較與產出驗證 |
| [tests/test_distributed_training.py](tests/test_distributed_training.py) | 全域梯度、不平均分片、尾批與整合測試 |
| [scripts/run_ddp_gpu.sh](scripts/run_ddp_gpu.sh) | CUDA 裝置條件檢查與 NCCL 執行入口 |
| [FP64 比較結果](reports/ddp_cpu_reference/comparison.json) | CPU 數值參考結果 |
| [FP32 比較結果](reports/ddp_cpu_validation/comparison.json) | CPU 排序與數值差異紀錄 |
| [FP64 二程序訓練摘要](reports/ddp_cpu_reference/round_0_world_2/results.json) | 時間、設定與各階段選模 |
| [FP32 二程序訓練摘要](reports/ddp_cpu_validation/round_0_world_2/results.json) | 原有精度的訓練與選模 |
| [二程序步驟紀錄](reports/ddp_cpu_reference/round_0_world_2/steps_rank_1.csv) | 各批樣本數與耗時 |
| [二程序通訊紀錄](reports/ddp_cpu_reference/round_0_world_2/communication.csv) | Gloo all-reduce 取樣 |
| [FP64 二程序 checkpoint](reports/ddp_cpu_reference/round_0_world_2/neural_recommender.pt) | 已執行數值參考的模型 |
| [FP32 二程序 checkpoint](reports/ddp_cpu_validation/round_0_world_2/neural_recommender.pt) | 已執行排序核對的模型 |

DDP 與通訊後端的 API 定義，依 [PyTorch 2.6 分散式訓練官方文件](https://docs.pytorch.org/docs/2.6/distributed.html)，本文件的數值來自上述 CPU 執行結果。
