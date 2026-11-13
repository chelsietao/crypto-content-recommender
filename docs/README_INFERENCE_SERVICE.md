# 獨立神經推薦推論服務與 HTTP 效能測試

本版本將雙塔召回與神經排序器接到獨立 HTTP 服務，服務啟動時載入已訓練的 checkpoint，透過 API 提供推薦結果，並使用另一個程序發送請求，完成輸出一致性與延遲、吞吐量及錯誤率量測。

服務 1.1 的負載控制、等待期限與回應驗證更新，請見 [服務優化紀錄](README_INFERENCE_SERVICE_REVIEW.md)，本文保留初版服務的實作與量測紀錄。

## 1. 本次完成內容

| 元件 | 完成內容 |
|---|---|
| 推論服務 | 使用 FastAPI 與 Uvicorn，提供健康檢查、就緒檢查、推薦與 API 文件 |
| 模型載入 | 啟動時核對 checkpoint 的 SHA-256，載入雙塔與排序器，完成暖機後提供服務 |
| 推論流程 | 共用固定權重與內容向量，每次請求計算使用者的候選召回與推薦結果 |
| 請求驗證 | 限制使用者 ID、模型名稱、推薦數量、已讀排除設定與額外欄位 |
| 結果核對 | API 輸出與離線推薦核對內容 ID、順序與分數，涵蓋 80 位既有使用者 |
| 效能測試 | 對兩種模型測試並發數 1、4、16，每組執行 3 回合，保存 9,000 筆請求紀錄 |

服務使用單一 Uvicorn worker、單一 PyTorch CPU 運算執行緒與一個模型推論席位，HTTP 請求可並發抵達，模型計算透過鎖依序執行，請求共用的使用者歷史與權重保持固定。

## 2. 啟動與使用

服務依賴放在 `requirements/requirements-service.txt`，在專案根目錄執行下列指令即可啟動。

```bash
python -m pip install -r requirements/requirements-service.txt
python -m src.serve_neural --host 127.0.0.1 --port 8000 --torch-threads 1
```

服務綁定本機位址 `127.0.0.1`，啟動後可從 [API 文件](http://127.0.0.1:8000/docs) 檢視請求與回應格式，也可使用以下端點。

| 方法與路徑 | 功能 |
|---|---|
| `GET /health` | 回傳程序健康狀態 |
| `GET /ready` | 回傳已載入的模型、checkpoint 雜湊、內容數與執行設定 |
| `POST /recommend` | 依使用者與模型回傳推薦內容 |
| `GET /docs` | Swagger API 操作文件 |
| `GET /openapi.json` | 機器可讀取的 API 規格 |

```bash
curl http://127.0.0.1:8000/ready

curl -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"U0001","model":"v7_pytorch_two_tower","k":10,"exclude_seen":true}'
```

## 3. 推薦請求與回應

| 欄位 | 規則 |
|---|---|
| `user_id` | 必填字串，去除前後空白後長度為 1 至 128 個字元 |
| `model` | `v7_pytorch_two_tower` 或 `v8_pytorch_neural_ranker`，預設為雙塔 |
| `k` | 1 至 50 的整數，預設為 10 |
| `exclude_seen` | 布林值，預設為 `true` |

請求驗證會拒絕未知模型、範圍外的推薦數、字串型數字、布林型推薦數與額外欄位，錯誤請求回傳 HTTP 422，模型尚未就緒時回傳 HTTP 503。

成功回應包含使用者 ID、模型名稱、checkpoint 雜湊、訓練截止時間與推薦清單，每筆推薦包含內容 ID、標題、主題標籤與有限分數，未知使用者採用模型的空歷史特徵。

V7 依雙塔內積取得前 50 筆候選，再輸出指定數量的推薦，V8 使用相同候選流程，接著透過神經排序器產生推薦分數，內容不足時回傳可用結果。

## 4. 實測環境與方法

本次使用合成資料訓練產出的 checkpoint，內容清單有 199 筆，效能請求輪流採用 80 位既有使用者與 1 位新使用者，兩個模型都使用 `k=10` 與 `exclude_seen=true`。

| 設定 | 本次執行值 |
|---|---|
| CPU | Apple M5 Pro |
| 實體／邏輯核心 | 15 / 15 |
| 記憶體 | 24.0 GiB |
| 作業系統與架構 | macOS-26.6.2-arm64-arm-64bit |
| Python / PyTorch | 3.9.6 / 2.6.0 |
| FastAPI / Uvicorn | 0.115.12 / 0.34.2 |
| 運算裝置 | CPU |
| Uvicorn worker | 1 |
| PyTorch 運算執行緒 | 1 |
| 模型推論席位 | 1 |
| checkpoint 載入與暖機至就緒 | 1.351 秒，包含子程序啟動與就緒輪詢 |

負載產生器與服務運行於同一台電腦，經 loopback HTTP/1.1 傳輸並保留連線，每個並發工作完成一筆請求後再送出下一筆，每回合測量前各工作先送出 5 筆暖機請求，暖機資料不列入統計。

每組並發設定每回合量測 500 筆請求，共 3 回合，表中的百分位數由該組 1,500 筆成功請求合併計算，每秒成功請求量以成功數量除以三回合的測量時間總和計算。

延遲從客戶端建立 HTTP 請求開始，到接收回應、解析 JSON 與核對推薦結果結束，包含排隊與傳輸時間，錯誤統計涵蓋連線失敗、逾時、非成功狀態與推薦結果不一致，請求失敗時不重試。

## 5. HTTP 效能結果

| 模型          |   並發數 |   請求數 |   p50 ms |   p95 ms |   p99 ms |   成功請求／秒 | 錯誤率   |
|:--------------|---------:|---------:|---------:|---------:|---------:|---------------:|:---------|
| V7 雙塔召回   |        1 |     1500 |    1.451 |    1.73  |    1.938 |         671.51 | 0%       |
| V7 雙塔召回   |        4 |     1500 |    3.525 |    4.451 |    4.788 |        1129.46 | 0%       |
| V7 雙塔召回   |       16 |     1500 |   12.425 |   84.643 |  154.041 |         636.39 | 0%       |
| V8 神經排序器 |        1 |     1500 |    1.581 |    1.979 |    2.146 |         608.49 | 0%       |
| V8 神經排序器 |        4 |     1500 |    4.626 |    5.212 |    5.666 |         862.04 | 0%       |
| V8 神經排序器 |       16 |     1500 |   12.724 |   84.063 |  243.786 |         569.18 | 0%       |

本次 9,000 筆測量請求的錯誤數為 0，並發數 16 的雙塔與神經排序器 p99 分別為 154.041 與 243.786 ms，原始紀錄與三回合明細保留這項結果，量測範圍為上述硬體、固定請求數與並發設定。

服務程序在各組測量前後的 RSS 介於 324.20 與 325.66 MiB，這些數值為每組量測邊界的記憶體快照，完整值記錄於結果 JSON。

## 6. 重現效能量測

以下指令會啟動另一個 Uvicorn 程序，等待模型就緒、核對程序與 checkpoint 身分，再完成六組量測，結束後關閉該測試服務。

```bash
python -m src.benchmark_service \
  --requests 500 --concurrency 1 4 16 --repetitions 3
```

輸出保存在 `reports/service_benchmark/`，包含逐筆請求、各組摘要、環境與回合明細，以及三個結果檔案的 SHA-256，測量程式會在出現請求或結果驗證錯誤時以失敗狀態結束。

## 7. 測試與模型檢查

本次檢查沿用既有的時間邊界、標籤、候選限制、固定種子訓練與 checkpoint 測試，新增 18 項 HTTP 服務測試，完整測試通過 68 項，套件相依性檢查與神經模型產出驗證通過。

服務測試涵蓋單次模型載入、就緒資訊、160 組模型與使用者配對的匯出結果、13 種無效請求、未知使用者、已讀控制、並發請求一致性，以及 checkpoint 雜湊錯誤的拒絕處理。

```bash
python -m pytest -q
python -m src.run_neural --verify-only
python -m pip check
```

GitHub Actions 加入服務依賴、服務測試與 HTTP 冒煙測試，冒煙測試會對兩種模型以並發數 1 與 4 各送出 20 筆測量請求，核對推薦輸出並保存測試結果。

## 8. 程式與證據索引

| 檔案 | 內容 |
|---|---|
| [src/serve_neural.py](../src/serve_neural.py) | 服務啟動、模型載入、請求驗證與 HTTP 端點 |
| [src/benchmark_service.py](../src/benchmark_service.py) | 獨立程序與 HTTP 效能量測 |
| [tests/test_service.py](../tests/test_service.py) | 服務行為、資料核對與並發測試 |
| [requirements/requirements-service.txt](../requirements/requirements-service.txt) | 服務與量測依賴 |
| [結果摘要](../reports/service_benchmark/summary.csv) | 六組並發設定的延遲、吞吐量與錯誤率 |
| [完整結果](../reports/service_benchmark/results.json) | 硬體、版本、checkpoint、程式雜湊與三回合明細 |
| [逐筆紀錄](../reports/service_benchmark/requests.csv) | 9,000 筆請求的延遲與驗證結果 |
| [結果檔案雜湊](../reports/service_benchmark/manifest.json) | 三個結果檔案的 SHA-256 |
| [雙塔模型說明](README_PYTORCH.md) | 訓練、模型架構與推薦指標 |
| [雙塔優化紀錄](README_PYTORCH_REVIEW.md) | 推論計算與產出驗證的修正 |

模型的載入與清理放在應用程式的生命週期中，實作依循 [FastAPI Lifespan 官方文件](https://fastapi.tiangolo.com/advanced/events/)，供請求共用啟動時建立的模型。
