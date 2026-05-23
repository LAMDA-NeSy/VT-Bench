# VLMs Prediction

一个用于视觉语言模型（VLMs）多模态微调的统一接口。支持在自定义数据集上微调 **Qwen3-VL** 和 **Table-LLaVA** 模型。

## ✨ 功能特性

- 🚀 **简单易用**：一行代码启动微调任务
- 📦 **可移植**：将文件夹传给别人即可使用
- 🔧 **灵活配置**：支持自定义训练参数
- 🧪 **消融实验**：内置 MCR 消融实验支持

## 📦 安装与依赖

### 前置条件

1. 安装 LLaMA-Factory：
```bash
cd /path/to/LLaMA-Factory
pip install -e .
```

2. 确保有支持的 GPU 和 CUDA 环境

### 可移植性

本模块设计为**可移植**的：
- 所有路径都使用相对路径或用户 home 目录
- 默认输出目录：`~/vlms_outputs/`
- 默认缓存目录：`~/.cache/huggingface/`

你可以直接将 `vlms_prediction/` 文件夹复制给别人使用。

## 🗂️ 数据集准备

### 数据集目录

将数据集文件放在模块的 `datasets/` 目录下：

```
vlms_prediction/
└── datasets/           # 👈 数据集目录
    └── mydata/
        ├── train.jsonl
        └── ...
```

### 支持的目录结构

**方式1：子目录结构（推荐）**

```
datasets/
└── mydata/
    ├── train.jsonl              # 必需
    ├── val.jsonl                # 推荐
    ├── test.jsonl               # 推荐
    ├── test_image_only.jsonl    # MCR消融实验用
    └── test_table_only.jsonl    # MCR消融实验用
```

**方式2：扁平结构**

```
datasets/
├── mydata_train.jsonl
├── mydata_val.jsonl
└── mydata_test.jsonl
```

### JSONL 文件格式

每行是一个 JSON 对象：

```json
{
  "id": "sample_001",
  "image": "/absolute/path/to/image.jpg",
  "messages": [
    {
      "role": "user",
      "content": "<image>\n\n你的问题...\n\n| 列1 | 列2 |\n|-----|-----|\n| 值1 | 值2 |"
    },
    {
      "role": "assistant",
      "content": "answer:0"
    }
  ]
}
```

**注意**：`image` 字段需要是**绝对路径**，指向图像文件。

## 🚀 快速开始

### Python API

```python
from vlms_prediction import VLMs_prediction

# 完整多模态训练
VLMs_prediction(
    data="mydata",      # 数据集名称
    model="qwen3",      # 模型：qwen3 或 tablellava
    diagnosis="full"    # 模式：full 或 mcr
)

# 消融实验
VLMs_prediction(data="mydata", model="qwen3", diagnosis="mcr")

# 先测试配置（不实际运行）
VLMs_prediction(data="mydata", model="qwen3", dry_run=True)

# 自定义输出目录
VLMs_prediction(data="mydata", model="qwen3", output_dir="/my/output/path")
```

### 命令行

```bash
# 查看数据集目录
python -m vlms_prediction.core.runner --show-dataset-dir

# 查看可用数据集
python -m vlms_prediction.core.runner --list-datasets

# 查看可用模型
python -m vlms_prediction.core.runner --list-models

# dry-run 测试
python -m vlms_prediction.core.runner -d mydata -m qwen3 --dry-run

# 运行训练
python -m vlms_prediction.core.runner -d mydata -m qwen3 -D full
```

## 🤖 支持的模型

| 模型 | 别名 | 说明 |
|------|------|------|
| `qwen3-vl-8b-instruct` | `qwen3`, `qwen` | Qwen3 VL 8B（推荐） |
| `tablellava-7b` | `tablellava` | Table-LLaVA 7B（表格优化） |

## 🔬 诊断模式

| 模式 | 说明 |
|------|------|
| `full` | 完整多模态训练（图像 + 表格） |
| `mcr` | 消融实验：自动运行 image_only 和 table_only |

## ⚙️ 高级配置

### 自定义参数

```python
VLMs_prediction(
    data="mydata",
    model="qwen3",
    num_epochs=5,
    batch_size=2,
    learning_rate=1e-5,
    max_samples=1000,
    output_dir="/custom/path"
)
```

### 预设配置

| 预设 | 说明 |
|------|------|
| `debug` | 快速测试，100样本 |
| `small_dataset` | 小数据集，5 epochs |
| `large_dataset` | 大数据集，2 epochs |
| `low_memory` | 低显存，4bit量化 |

```python
VLMs_prediction(data="mydata", model="qwen3", preset="debug")
```

### 环境变量

可通过环境变量自定义缓存目录：

```bash
export HF_HOME=/path/to/cache
export TMPDIR=/path/to/tmp
python run_vlms.py
```

## 📁 目录结构

```
vlms_prediction/
├── __init__.py
├── README.md
├── datasets/           # 数据集目录
│   └── mydata/
│       ├── train.jsonl
│       └── ...
├── config/
│   ├── datasets.py     # 数据集自动扫描
│   ├── models.py       # 模型配置
│   └── base_config.py  # 训练参数
└── core/
    └── runner.py       # 核心逻辑
```

## 📤 输出目录

默认输出到 `~/vlms_outputs/`：

```
~/vlms_outputs/{dataset}/{model}_{diagnosis}/
├── train_config.yaml
├── dataset_info.json
├── adapter_model/
└── ...
```

## ❓ FAQ

**Q: 如何将模块分享给别人？**

复制整个 `vlms_prediction/` 文件夹即可。别人需要：
1. 安装 LLaMA-Factory
2. 准备自己的数据集放入 `datasets/` 目录

**Q: 如何使用单个 GPU？**
```bash
CUDA_VISIBLE_DEVICES=0 python run_vlms.py
```

**Q: 如何修改默认输出目录？**
```python
VLMs_prediction(data="mydata", model="qwen3", output_dir="/my/path")
```

## 📝 License

遵循 LLaMA-Factory 开源协议。
