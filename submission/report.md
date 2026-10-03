# Day16 Track 2 — Cloud AI Environment Setup

1. Cloud/region/instance: AWS us-east-1, CPU benchmark trên t3.small; source commit: 55539f67d7c78b43afe334a2ec3271c4bfdbbe2.
2. Dataset: Credit Card Fraud Detection, 284,807 rows, stratified 80/20 split, random_state=42.
3. Benchmark measured data loading time and LightGBM training time.
4. Evaluation metrics: AUC-ROC, Accuracy, F1, Precision, Recall.
5. Inference benchmark measured single-row latency and throughput on a 1,000-row batch.
6. CPU/RAM/network observations were collected after the benchmark; the screenshot records the resource state at that time.
7. AWS Billing was checked during the lab; billing data had not fully updated at the time of observation.
8. benchmark.py and benchmark_result.json were downloaded from the VM before infrastructure cleanup.