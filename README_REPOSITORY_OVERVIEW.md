# Crypto Content Recommender：系統總覽與驗證摘要

這個專案以加密貨幣內容推薦為情境，串接合成資料、時間切分、九種推薦方法、離線品質評估、展示介面、HTTP 推論服務，以及 GPU profiling 與分散式訓練驗證，模型與量測結果均保留對應的資料及驗證入口。

## 本次 Transformer 複查

本次檢查涵蓋曝光標籤的時間範圍、V8 與 V9 的共同候選樣本、Transformer padding 遮罩、checkpoint 綁定、推薦品質、延遲統計及資源單位，另以相同設定重訓 Transformer，核對每個參數、完整訓練曲線與樣本時間紀錄。

重訓後的最大參數差異為 0，曲線、選模結果與樣本紀錄一致，推論批次大小 1、7、50 通過分數核對，所有內容皆已看過的情況回傳空推薦，既有 12 個產出檔案、81 位使用者推薦與 800 筆效能紀錄通過驗證，本次檢查未重現模型或既有量測結果錯誤。

本次重訓與稽核紀錄保存於 [TRANSFORMER_AUDIT.json](reports/TRANSFORMER_AUDIT.json)，Transformer 的品質、延遲與資源量測口徑見 [版本說明](README_TRANSFORMER_RERANKING.md)。

完整本機測試共 198 項通過，包含 CPU 雙程序 DDP、關閉 CPU fallback 的 MPS 檢查、Transformer 時間區段及產出修改測試，本次保留原有模型參數與效能量測資料。

## 資料與評估

資料包含 240 筆內容、80 位合成使用者與 8,564 筆互動，互動類型包含曝光、點擊、喜歡、收藏與隱藏，共用截止點為 `2026-03-08T04:48:00+00:00`，訓練保留 6,852 筆事件及截止點前已發布的 199 筆內容。

測試目標取自截止點後的正向互動，排除訓練期間已見內容與含隱藏回饋的配對，共有 30 位使用者符合條件，本次彙整已核對九個模型使用相同的使用者及目標內容。

V7 雙塔先使用前段時間區段訓練與選模，權重凍結後供 V8 與 V9 建立候選，兩個排序器使用後段區段的曝光樣本，各含 533 筆訓練樣本與 394 筆驗證樣本，特徵取自請求前的歷史，曝光標籤觀察範圍為 15 分鐘且限制於所屬區段。

## 九個模型與品質結果

下表依每位使用者前 10 筆推薦計算，V1 至 V6 採用各自的候選規則，V7 至 V9 共用上限 50 筆的雙塔候選，數值記錄各推薦流程的結果。

| 模型 | 完成內容 | Hit / Recall@10 | NDCG@10 | 候選召回率 |
|---|---|---:|---:|---:|
| V1 熱門推薦 | 互動權重與時間衰減 | 0.1333 | 0.0525 | 1.0000 |
| V2 內容推薦 | TF-IDF 與使用者偏好 | 0.1333 | 0.0711 | 1.0000 |
| V3 協同過濾 | 稀疏內容相似鄰居 | 0.0333 | 0.0105 | 1.0000 |
| V4 混合召回 | 熱門、內容、協同與探索通道 | 0.1333 | 0.0617 | 1.0000 |
| V5 邏輯迴歸排序 | 召回候選的特徵排序 | 0.6667 | 0.3981 | 1.0000 |
| V6 多樣性重排 | 來源及主題上限 | 0.6667 | 0.3991 | 1.0000 |
| V7 PyTorch 雙塔 | 使用者塔與內容塔的向量召回 | 0.1333 | 0.0452 | 0.7333 |
| V8 神經排序器 | 凍結雙塔後的候選特徵排序 | 0.2333 | 0.1013 | 0.7333 |
| V9 Transformer | 使用者向量與內容詞元的交互注意力 | 0.2000 | 0.1225 | 0.7333 |

這組合成資料的最高 NDCG@10 為 V6 的 0.3991，V9 的 NDCG@10 為 0.1225、命中率為 0.2000，其相對 V8 的逐位使用者 NDCG 差值平均為 0.0212，95% bootstrap 區間為 −0.0893 至 0.1458，完整彙整見 [九模型指標 CSV](reports/REPOSITORY_MODEL_METRICS.csv)。

## 展示、服務與硬體實測

| 元件 | 已完成範圍 | 結果與入口 |
|---|---|---|
| Streamlit | V1 至 V6 互動推薦、回饋與儀表板，V7 與 V8 的已儲存推薦、指標及訓練曲線 | [公開展示](https://rkatmcugsybuh4d5cejg6s.streamlit.app/)，程式位於 `app.py` |
| CPU HTTP 服務 | V7 與 V8 的 FastAPI 推論、嚴格格式驗證、容量上限及鎖等待期限 | [推論服務 1.1](README_INFERENCE_SERVICE_REVIEW.md)，保存 9,000 筆 HTTP 請求的量測 |
| Apple GPU profiling | V7 與 V8 的 MPS 輸入、批次及排序路徑，六回合輪替與推薦一致性核對 | [GPU 量測方法](README_GPU_MEASUREMENT_AUDIT.md)，裝置為 Apple M5 Pro 的 MPS |
| DDP 訓練 | CPU／Gloo 一程序與二程序，固定全域 batch、梯度加權、checkpoint 及通訊取樣 | [DDP 訓練](README_DDP_TRAINING.md)，FP64 與 FP32 各保存一份比較報告 |
| Transformer 量測 | V8 與 V9 的 CPU 函式延遲、吞吐量、CPU 時間、取樣 RSS 與參數大小 | [Transformer 重排序](README_TRANSFORMER_RERANKING.md)，四回合共 800 筆請求 |

DDP 程式提供 CUDA／NCCL 分支，repo 保存的分散式訓練實測為 CPU／Gloo，兩份報告均標記 `gpu_result: false`，每份覆蓋 23 個檔案與 1,440 個全域訓練步驟。

Transformer 採 CPU FP32、單一運算執行緒及單請求併發，V8 的 p95 為 0.4186 毫秒，V9 為 3.9040 毫秒，V9 每 1,000 次請求使用 3.3439 CPU 秒、取樣峰值 RSS 為 311.97 MiB，數值對應暖機後的程序內推薦函式，HTTP 服務量測則包含傳輸、排隊及回應驗證。

## 產出與追溯

| 目錄或檔案 | 內容 |
|---|---|
| `data/processed` | 可重現的合成內容、使用者與互動資料 |
| `artifacts` | V1 至 V6 指標、推薦、未命中紀錄與執行摘要 |
| `artifacts/neural` | 雙塔與 V8 checkpoint、訓練樣本、曲線及八模型評估 |
| `reports/service_review_benchmark` | HTTP 請求原始紀錄與效能統計 |
| `reports/gpu_profile_audit` | MPS 原始計時、模式順序、記憶體及配對吞吐量 |
| `reports/ddp_cpu_reference` | FP64 訓練的一程序與二程序數值比較 |
| `reports/ddp_cpu_validation` | FP32 訓練的一程序與二程序推薦排序比較 |
| `reports/transformer_reranking` | V9 checkpoint、時間稽核、品質、延遲與資源成本 |
| `.github/workflows/verify.yml` | 基礎流程重現、模型測試、產出驗證與 HTTP smoke test |

模型及量測目錄使用 manifest 保存產出 SHA-256，驗證入口會依各版本核對 checkpoint、推薦或原始量測統計，歷次問題與修正保留在各版本 README。
