# Crypto Content Recommendation System

這個專案以加密貨幣內容推薦為主題，完成合成資料產生、六種推薦基線、PyTorch 雙塔召回與神經排序器、Transformer 內容重排序、離線評估及 Streamlit 展示，並提供 CPU HTTP 推論服務、Apple GPU profiling 與 DDP 訓練驗證。

完整架構、九模型結果、展示與服務涵蓋範圍，以及 Transformer 重訓複查，請見 [系統總覽與驗證摘要](README_REPOSITORY_OVERVIEW.md)。

三項整合改進已接通 V9 HTTP、九模型展示及評估重建，請見 [本次優化說明](README_SYSTEM_INTEGRATION.md)。

## 資料與評估

資料包含 240 筆內容、80 位合成使用者與 8,564 筆互動，互動類型包括曝光、點擊、喜歡、收藏與隱藏，使用固定亂數種子產生。

評估採用共用時間截止點，訓練使用截止點前的 6,852 筆互動與當時已發布的 199 筆內容，測試使用後續正向互動，排除已看過的內容與含隱藏回饋的配對，共有 30 位使用者符合條件。

| 模型 | 方法 | Hit / Recall@10 | NDCG@10 |
|---|---|---:|---:|
| V1 | 熱門推薦與時間衰減 | 0.1333 | 0.0525 |
| V2 | TF-IDF 內容推薦 | 0.1333 | 0.0711 |
| V3 | 稀疏鄰居協同過濾 | 0.0333 | 0.0105 |
| V4 | 四通道混合召回 | 0.1333 | 0.0617 |
| V5 | 邏輯迴歸排序 | 0.6667 | 0.3981 |
| V6 | 來源與主題多樣性重排 | 0.6667 | 0.3991 |
| V7 | PyTorch 雙塔召回 | 0.1333 | 0.0452 |
| V8 | 凍結雙塔後的神經排序器 | 0.2333 | 0.1013 |
| V9 | Transformer 內容重排序 | 0.2000 | 0.1225 |

九個模型的使用者與測試目標一致，V1 至 V6 採各自候選規則，V7 至 V9 共用最多 50 筆雙塔候選，完整指標見 [九模型 CSV](reports/REPOSITORY_MODEL_METRICS.csv)，逐位使用者結果保存在 `artifacts` 與 `reports/transformer_reranking`。

## 展示與推論

[Streamlit 公開展示](https://rkatmcugsybuh4d5cejg6s.streamlit.app/) 提供 V1 至 V6 的互動推薦、工作階段回饋及儀表板，`Saved Model Results` 分頁展示九模型指標、候選數量及品質區間，並提供 V7、V8、V9 的已儲存推薦與訓練曲線。

獨立 FastAPI 服務 1.2 提供 V7、V8、V9 的 CPU 推論，實作雙 checkpoint 身分驗證、容量控制及鎖等待期限，最新 HTTP 效能資料保存於 `reports/service_v12_benchmark`，啟動與量測方式見 [整合說明](README_SYSTEM_INTEGRATION.md)。

V9 保存 Transformer checkpoint、訓練樣本時間稽核及品質評估，CPU 延遲、吞吐量與資源成本採四回合、八個獨立工作程序完成量測，詳見 [Transformer 重排序](README_TRANSFORMER_RERANKING.md)。

## 硬體與分散式訓練

GPU profiling 使用 Apple M5 Pro 的 MPS，核對 V7 與 V8 的批次推論、推薦一致性與排序優化，六回合輪替結果見 [GPU 量測方法](README_GPU_MEASUREMENT_AUDIT.md)。

DDP 保存 CPU／Gloo 一程序與二程序的實測結果，涵蓋全域 batch、梯度加權、checkpoint 重載及通訊取樣，程式另提供 CUDA／NCCL 分支，說明見 [分散式訓練](README_DDP_TRAINING.md)。

## 執行方式

本機驗證使用 Python 3.9 與 PyTorch 2.6.0，GitHub Actions 使用 Python 3.11，安裝完整依賴後可執行測試、基礎資料流程及展示介面。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-service.txt
RUN_DDP_INTEGRATION=1 PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -q
python -m src.run_pipeline
streamlit run app.py
```

核對已保存的 Transformer 與 DDP 產出時，使用以下命令，推薦與品質會從 checkpoint 重算，效能統計則由原始量測紀錄重算。

```bash
python -m src.evaluate_repository --verify-only
python -m src.run_transformer --verify-only
python -m src.validate_distributed --verify-only --output reports/ddp_cpu_reference
python -m src.validate_distributed --verify-only --output reports/ddp_cpu_validation
```

## 驗證與版本文件

本次本機完整測試共 214 項通過，涵蓋九模型重建、Streamlit 操作、三模型 HTTP、CPU 雙程序 DDP 與關閉 CPU fallback 的 MPS 檢查，記錄見 [整合驗證](reports/SYSTEM_INTEGRATION_VALIDATION.json)，Transformer 重訓的最大參數差異為 0，記錄見 [重訓摘要](reports/TRANSFORMER_AUDIT.json)。

| 主題 | 文件 |
|---|---|
| 三項整合改進 | [服務、展示與評估](README_SYSTEM_INTEGRATION.md) |
| 全專案範圍與九模型結果 | [系統總覽](README_REPOSITORY_OVERVIEW.md) |
| 時間切分與六種基線修正 | [系統修正報告](reports/SYSTEM_REVIEW_AND_FIXES.md) |
| 雙塔與神經排序器 | [PyTorch 實作](README_PYTORCH.md)、[召回優化](README_PYTORCH_REVIEW.md) |
| HTTP 推論與容量控制 | [服務量測](README_INFERENCE_SERVICE.md)、[服務 1.1](README_INFERENCE_SERVICE_REVIEW.md) |
| Apple GPU profiling | [profiling 實作](README_GPU_PROFILING.md)、[候選上限修正](README_GPU_PROFILING_REVIEW.md)、[輪替量測](README_GPU_MEASUREMENT_AUDIT.md) |
| DDP | [訓練實作](README_DDP_TRAINING.md)、[步驟與程序清理](README_DDP_REVIEW.md)、[曲線驗證](README_DDP_HISTORY_REVIEW.md) |
| Transformer | [品質、延遲及資源成本](README_TRANSFORMER_RERANKING.md) |
