# 推薦系統整合：Transformer 服務、九模型評估與展示

本次將 V9 Transformer 接入 HTTP 推論及 Streamlit 展示，並新增九模型評估重建入口，沿用既有模型權重與時間切分，讓模型結果、服務輸出與展示資料連接同一組 checkpoint 及評估產出。

## 三項改進

| 項目 | 原有狀態 | 完成內容 |
|---|---|---|
| Transformer 服務整合 | HTTP 服務提供 V7 與 V8，V9 透過 Python 程式執行 | 服務更新為 1.2.0，提供 V7、V8、V9 推論，共用已載入的雙塔與內容向量 |
| 展示整合 | 展示頁保存 V7、V8 結果，V9 報告位於 repo | 新增九模型結果、候選規則、品質差值區間，以及三個神經模型的使用者推薦選單 |
| 評估自動化 | 九模型總表由不同版本結果彙整 | 新增單一命令，從資料及 checkpoint 重建九模型評估、逐位使用者結果、候選數量與品質區間 |

## 1. HTTP 推論服務 1.2

`POST /recommend` 接受 `v9_transformer_content_reranker`，三個模型共用一個 NeuralSystem，基礎 checkpoint 在服務啟動時載入一次，V9 載入自己的排序權重並共用使用者特徵與內容向量。

兩份 checkpoint 均先核對相鄰 manifest 的 SHA-256，V9 保存的基礎 checkpoint 雜湊也須一致，啟動失敗時服務不進入 ready 狀態。

回應保留 `checkpoint_sha256` 作為基礎模型身分，新增 `reranker_checkpoint_sha256` 表示 V9 權重，V7 與 V8 的此欄位為 `null`，`GET /ready` 列出三個模型及 Transformer checkpoint 雜湊，負載測試會核對兩份模型身分、推薦內容、順序與分數。

原有容量上限、單一推論席位與鎖等待期限維持運作，`--disable-transformer` 可啟動 V7、V8 服務，此時 V9 請求回傳 HTTP 422。

```bash
python -m src.serve_neural --port 8000
```

```bash
curl http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"U0001","model":"v9_transformer_content_reranker","k":10}'
```

### HTTP 實測

本次使用獨立 Uvicorn 程序，在 Apple M5 Pro、24 GiB 記憶體、CPU 單一運算執行緒環境執行，每個模型分別以併發 1、4 測試，各兩回合、每回合 100 筆，共完成 1,200 次量測請求，錯誤數為 0。

| 模型 | 併發數 | 請求數 | p50 ms | p95 ms | p99 ms | 成功請求／秒 |
|---|---:|---:|---:|---:|---:|---:|
| V7 | 1 | 200 | 1.784 | 2.152 | 2.250 | 549.96 |
| V7 | 4 | 200 | 4.094 | 5.331 | 5.643 | 967.06 |
| V8 | 1 | 200 | 1.615 | 1.939 | 2.045 | 604.69 |
| V8 | 4 | 200 | 4.669 | 5.315 | 5.919 | 836.46 |
| V9 | 1 | 200 | 5.195 | 6.051 | 6.321 | 189.68 |
| V9 | 4 | 200 | 16.878 | 17.742 | 18.118 | 238.06 |

延遲包含客戶端請求建立、HTTP 傳輸、排隊、JSON 解析與結果驗證，每組測試先對每個併發工作暖機五次，暖機不列入統計，原始請求、逐回合數據、硬體設定與摘要保存於 [service_v12_benchmark](../reports/service_v12_benchmark/)。

```bash
python -m src.benchmark_service --requests 100 --concurrency 1 4 \
  --repetitions 2 --output /tmp/service-v12-reproduction
```

## 2. Streamlit 九模型展示

[公開展示](https://rkatmcugsybuh4d5cejg6s.streamlit.app/) 的 `Saved Model Results` 分頁提供九模型品質表、候選數量、相對 V6 的品質區間，以及 V7、V8、V9 的使用者推薦清單，使用者選單包含 80 位既有使用者與一位新使用者，並可下載九模型指標 CSV。

選擇 V9 時顯示 Transformer 訓練曲線，選擇 V7 或 V8 時顯示雙塔與神經排序器的曲線，六種基線的互動推薦與工作階段回饋保留在 `User Feed`。

展示資料由 `src/portfolio_exports.py` 讀取，先核對六個產出檔案的 manifest 與雜湊，再檢查九模型及使用者覆蓋，資料錯誤會顯示驗證訊息，展示讀取器使用 NumPy 與 pandas，無須匯入 PyTorch 或載入模型權重。

## 3. 九模型評估重建

`src/evaluate_repository.py` 由固定合成資料與共同時間切分開始，訓練六種基線、載入經驗證的 V7、V8、V9 checkpoint，再重新計算九模型的品質與推薦，輸出目錄要求為空目錄。

```bash
python -m pip install -r requirements/requirements-service.txt
python -m src.evaluate_repository --output reports/repository_evaluation_new
python -m src.evaluate_repository --verify-only
```

每個模型使用同一組 30 位使用者及測試目標，候選池規則另列保存，避免將不同召回範圍混為同一設定。

| 模型群組 | 本次每位測試使用者的候選數量 |
|---|---|
| V1、V2、V3 | 最少 137，平均 141.2，最多 145 |
| V4、V5、V6 | 最少 61，平均 65.97，最多 70 |
| V7、V8、V9 | 每人 50 筆，三個模型共用雙塔候選 |

品質區間以 V6 為參照，每個模型分別計算逐位使用者的 Hit 與 NDCG 差值，共產生 18 列統計，採用固定亂數種子的 2,000 次使用者配對 bootstrap，V6 自身的差值與區間均為 0。

評估產出保存於 [repository_evaluation](../reports/repository_evaluation/)，包含 9 列模型指標、270 列逐位使用者結果、243 組神經模型推薦、候選規則、品質區間與執行摘要，`--verify-only` 會從模型重新計算並核對全部表格。推薦 ID 與順序逐筆比對，分數沿用推論驗證的絕對容差 1e-6 與相對容差 1e-5，以容納 macOS 與 Linux 的 FP32 捨入差異。

## 驗證與修改檔案

完整本機回歸共 214 項通過，包含新增 16 項整合測試，以及 CPU 雙程序 DDP 與關閉 CPU fallback 的 MPS 檢查。

測試涵蓋三個模型的服務身分、V9 全部 81 位使用者的 HTTP 推薦、共用模型物件、Transformer 停用與雜湊錯誤，以及九模型重建、品質區間、候選資料及請求缺漏的拒絕情境。

Streamlit 操作測試會開啟展示、選擇 V9 與新使用者，核對九模型表及推薦內容，展示讀取器另由獨立 Python 程序驗證沒有匯入 PyTorch。

| 檔案 | 職責 |
|---|---|
| `src/serve_neural.py` | 三模型推論、雙 checkpoint 身分與版本 1.2.0 |
| `src/benchmark_service.py` | 三模型真實 HTTP 量測與回應核對 |
| `src/transformer_reranker.py` | 將 V9 權重接至服務共用的基礎系統 |
| `src/checkpoint_io.py` | checkpoint 與相鄰 manifest 的雜湊驗證 |
| `src/evaluate_repository.py` | 九模型重建、候選統計與配對品質區間 |
| `src/portfolio_exports.py` | 不載入權重的展示資料驗證與讀取 |
| `app.py` | 九模型及 V9 推薦展示 |

GitHub Actions 納入九模型重建及驗證，基礎工作執行展示操作測試，神經模型工作執行三模型服務與評估回歸，測試與量測摘要見 [整合驗證紀錄](../reports/SYSTEM_INTEGRATION_VALIDATION.json)。
