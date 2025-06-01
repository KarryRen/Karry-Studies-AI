# 基于 CNN 的高频交易噪声过滤与多空信号识别

任凯（2024 金融科技，2401212437）

## 1. Introduction

本作业以 Kaggle 竞赛 Jane Street 数据为例，展示 CNN 在时序数据处理中的特征提取与降噪方法，理解高频交易数据的特性与挑战。作业的核心是为了培养金融工程思维：建立从数据清洗、特征工程到模型构建与训练的完整量化研究流程，初步具备构建金融智能策略系统的能力。本报告详细说明了作者解决作业问题的整体思路。

## 2. Data Clean

### 2.1 Dataset Description

本作业数据集来自 Kaggle 上 Jane Street 发起的一个[**竞赛**](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview)。此数据库核心是匿名处理的真实交易数据，但由于比赛限制我们无法拿到合理的 test 数据针对真实情况进行调优，因此本作业解答专注在对 train 数据集上的研究，不调用主办方提供的测试数据和接口，以实现更加完整、流畅的练习目的，并呈现相对清晰、泛化的解题思路和框架。

在此我对训练数据 `train.parquet` 进行了详细描述。

首先，在竞赛主页上完成原始数据下载后，训练数据的路径结构如下，由于整体数据量过大，所以将其分成十个部分（`id` 从 0 到 9），并以 `parquet` 的文件类型进行存储，**总内存约占 12.8 GB**。

```python
train.parquet/
├── partition_id=0
    └── part-0.parquet
├── partition_id=1
    └── part-1.parquet
├── ...
└── partition_id=9
    └── part-9.parquet
```





## Model Construction

## Train & Valuation







## References

- https://mp.weixin.qq.com/s?__biz=MzIwNDA5NDYzNA==&mid=2247507263&idx=1&sn=08fdc369d3c85949c8dfd4b89d5247ca&chksm=979609982450787a71da1a48b9998a44e1b179cf1a82ea7bd0076fc21f0df5c6a259011eb9b3#rd
- https://blog.csdn.net/weixin_48152827/article/details/144516229
- https://github.com/evgeniavolkova/kagglejanestreet/blob/master/scripts/python/run_full.py
- https://www.kaggle.com/code/yuanzhezhou/jane-street-baseline-lgb-xgb-and-catboost

