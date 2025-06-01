# 基于 CNN 的高频交易噪声过滤与多空信号识别

任凯（2024 金融科技，2401212437）

> 项目结构如下
>
> ```python
> JS-Real-Time-Market-Data-Forecasting/
> ├── Data/ # Data directory, too big to upload.
> ├── Code/ # Code directory
>     ├── data_preprocess/ # code for data preprocessing
>         ├── utils.py # Utils functions for data preprocessing.
> └── partition_id=9
> ```

## 1. Introduction

本作业以 Kaggle 竞赛 Jane Street 数据为例，展示 CNN 在时序数据处理中的特征提取与降噪方法，理解高频交易数据的特性与挑战。作业的核心是为了培养金融工程思维：建立从数据清洗、特征工程到模型构建与训练的完整量化研究流程，初步具备构建金融智能策略系统的能力。本报告详细说明了作者解决作业问题的整体思路。

## 2. Data Clean

### 2.1 Dataset Description

本作业数据集来自 Kaggle 上 Jane Street 发起的一个[**竞赛**](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview)。此数据库核心是匿名处理的真实交易数据，但由于比赛限制我们无法拿到合理的 test 数据针对真实情况进行调优，因此本作业解答专注在对 train 数据集上的研究，不调用主办方提供的测试数据和接口，以实现更加完整、流畅的练习目的，并呈现相对清晰、泛化的解题思路和框架。在此我对训练数据 `train.parquet` 进行了如下详细描述。

首先，在竞赛主页上完成**原始数据**下载后，训练数据的路径结构如下，由于整体数据量过大，所以将其分成十个部分（`id` 从 0 到 9），并以 `parquet` 的文件类型进行存储，**总内存约占 12.3 GB**。考虑到原始数据量过大，后续处理过于耗时，本项目只使用了 `id` 从 0 到 4 的 5 组数据，其中共包含 850 日下的 16,825,066 条交易数据，共出现 33 种交易标的。

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

其次，读取数据并结合官方描述，可以得到对**数据字段**更详细的说明。共有 92 个字段，可分为如下 4 类：

- 标识性字段（3 个）
  - `date_id` 和 `time_id`： 按顺序排序的整数值，为数据提供时间顺序结构，尽管 `time_id` 值之间的实际时间间隔可能有所不同
  - `symbol_id`：匿名化的标的代码，唯一标识
- 权重字段（1 个）
  - `weight`：用于计算评分函数的权重
- 特征字段（79 个）
  - `feature_00` 到 `feature_78`：共 79 个匿名市场特征数据
- 响应字段（9 个）
  - `responder_0` 到 `responder_8`：共 9 个匿名响应者，介于 -5 和 5 之间。`responder_6` 是预测目标。

总的来看，数据集中的每一行都对应一个标的（用标识 `symbol_id` 标识）和一个时间戳（用 `date_id`和 `time_id` 标识）的唯一组合。`date_id` 列是一个整数，表示事件发生的日期，而 `time_id` 表示时间顺序。需要注意的是，它们之间的实际时间差异 `time_id` 不能保证一致。**注意**：`symbol_id` 字段实际是加密标识符，但 `symbol_id` 不保证每个标识符都会出现在所有 `time_id` 和 `date_id` 组合中。

### 2.2 Data Preprocess

原始数据存在众多问题，无法对其直接建模，需要进行多步骤的预处理。

**Step 1. 减少数据占存：筛选目标列，修改数据类型**

一方面，在响应字段中只保留预测目标 `responder_6`，这样总字段数变为 84 个。另一方面，对剩余的每个字段进行数据类型规范化，通过修改数据类型以减少大约 48.6% 的占存（从 5150.65 MB 到 2647.53 MB）。

**Step 2. 处理空缺值**

正如数据描述中的那样，不同日期下的 `symbol_id` 情况不同，一些标的会在某些日期中缺失。通过图 1. 可以发现，在 850 个交易日里，500 天前交易日中的 `symbol_id` 较比全局存在较多缺省，这对整体建模是十分不利的，因此选择直接剔除。剩余 350 个交易日下共 9,274,909 条交易数据。

紧接着需要对每个特征的空值进行检测，如表 1. 所呈现的对每一个建模特征和指标的描述性统计显示，大部分都存在空值，可以直接按照 `symbol_id` 分组前填充即刻



## Model Construction

## Train & Valuation







## References

- https://mp.weixin.qq.com/s?__biz=MzIwNDA5NDYzNA==&mid=2247507263&idx=1&sn=08fdc369d3c85949c8dfd4b89d5247ca&chksm=979609982450787a71da1a48b9998a44e1b179cf1a82ea7bd0076fc21f0df5c6a259011eb9b3#rd
- https://blog.csdn.net/weixin_48152827/article/details/144516229
- https://github.com/evgeniavolkova/kagglejanestreet/blob/master/scripts/python/run_full.py
- https://www.kaggle.com/code/yuanzhezhou/jane-street-baseline-lgb-xgb-and-catboost

