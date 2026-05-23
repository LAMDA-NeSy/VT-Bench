#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLMs Prediction 运行脚本
========================

使用前请先将数据集放入 src/vlms_prediction/datasets/ 目录

数据集文件结构:
    datasets/
    └── mydata/
        ├── train.jsonl      (必需)
        ├── val.jsonl        (推荐)
        ├── test.jsonl       (推荐)
        ├── test_image_only.jsonl  (消融实验用)
        └── test_table_only.jsonl  (消融实验用)

详细说明请参考: src/vlms_prediction/README.md
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vlms_prediction import VLMs_prediction
from vlms_prediction.core.runner import list_datasets, list_models, show_dataset_dir


# ============================================================
# 配置区域 - 修改这里的参数来运行实验
# ============================================================

# 数据集名称（对应 datasets/ 目录下的子目录名）
DATA = "mydata"  # 修改为你的数据集名称

# 模型选择
#   - "qwen3": Qwen3 VL 8B（推荐）
#   - "tablellava": Table-LLaVA 7B
MODEL = "qwen3"

# 诊断模式
#   - "full": 完整多模态训练（图像+表格）
#   - "mcr":  消融实验（自动运行 image_only 和 table_only）
DIAGNOSIS = "full"

# 是否只打印配置不实际运行
DRY_RUN = True  # 改为 False 开始实际训练

# 预设配置 (可选: None, "debug", "small_dataset", "large_dataset", "low_memory")
PRESET = None

# 自定义训练参数 (设为 None 使用默认值)
NUM_EPOCHS = None
BATCH_SIZE = None
LEARNING_RATE = None
MAX_SAMPLES = None
OUTPUT_DIR = None


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VLMs Prediction")
    print("=" * 60)
    print(f"数据集目录: {show_dataset_dir()}")
    print()
    
    # 显示可用数据集
    available_datasets = list_datasets()
    print("可用数据集:")
    if available_datasets:
        for name in available_datasets:
            print(f"  - {name}")
    else:
        print("  (无数据集)")
        print()
        print("请将数据集放入以下目录:")
        print(f"  {show_dataset_dir()}/")
        print()
        print("目录结构示例:")
        print("  mydata/")
        print("    ├── train.jsonl  (必需)")
        print("    ├── val.jsonl    (推荐)")
        print("    └── test.jsonl   (推荐)")
        print()
        print("详细说明请参考: src/vlms_prediction/README.md")
        sys.exit(1)
    
    print()
    print("可用模型:")
    for name, desc in list_models().items():
        print(f"  - {name}: {desc}")
    print()
    
    # 检查数据集是否存在
    if DATA not in available_datasets:
        print(f"❌ 错误: 数据集 '{DATA}' 不存在")
        print(f"   请将数据集放入: {show_dataset_dir()}/{DATA}/")
        sys.exit(1)
    
    print("-" * 60)
    print(f"当前配置:")
    print(f"  数据集: {DATA}")
    print(f"  模型: {MODEL}")
    print(f"  诊断模式: {DIAGNOSIS}")
    print(f"  Dry Run: {DRY_RUN}")
    print("-" * 60)
    print()
    
    # 运行
    result = VLMs_prediction(
        data=DATA,
        model=MODEL,
        diagnosis=DIAGNOSIS,
        dry_run=DRY_RUN,
        preset=PRESET,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_samples=MAX_SAMPLES,
        output_dir=OUTPUT_DIR,
    )
    
    print("\n运行结果:")
    print(f"  状态: {result['status']}")
    print(f"  实验: {result['experiments']}")
    print(f"  输出目录: {result['output_dirs']}")
