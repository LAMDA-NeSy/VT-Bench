# 数据集配置模块
# 支持两种目录结构:
#   1. 子目录结构: datasets/{name}/train.jsonl
#   2. 扁平结构: datasets/{name}_train.jsonl

import os
from typing import TypedDict, Optional
from pathlib import Path


class DatasetConfig(TypedDict):
    """数据集配置类型"""
    train: str              # 训练集文件路径
    valid: str              # 验证集文件路径
    test: str               # 测试集文件路径
    test_image_only: str    # 仅图像测试集（消融实验用）
    test_table_only: str    # 仅表格测试集（消融实验用）
    output_base_dir: str    # 输出目录基础路径
    description: str        # 数据集描述


# 数据集目录路径
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# 默认输出目录（使用用户 home 目录）
DEFAULT_OUTPUT_BASE = os.path.join(os.path.expanduser("~"), "vlms_outputs")


def _find_dataset_file(dataset_name: str, file_type: str) -> Optional[str]:
    """
    查找数据集文件，支持两种命名方式
    """
    # 方式1: 子目录结构 datasets/{name}/{type}.jsonl
    subdir_path = DATASETS_DIR / dataset_name / f"{file_type}.jsonl"
    if subdir_path.exists():
        return str(subdir_path)
    
    # 方式2: 扁平结构 datasets/{name}_{type}.jsonl
    flat_path = DATASETS_DIR / f"{dataset_name}_{file_type}.jsonl"
    if flat_path.exists():
        return str(flat_path)
    
    return None


def _get_dataset_files(dataset_name: str) -> dict[str, str]:
    """
    获取数据集的所有文件路径
    """
    files = {}
    
    # 训练集
    train = _find_dataset_file(dataset_name, "train")
    if train:
        files["train"] = train
    
    # 验证集 (支持 val 和 valid 两种命名)
    valid = _find_dataset_file(dataset_name, "val")
    if not valid:
        valid = _find_dataset_file(dataset_name, "valid")
    if valid:
        files["valid"] = valid
    
    # 测试集
    test = _find_dataset_file(dataset_name, "test")
    if test:
        files["test"] = test
    
    # 消融实验数据集
    image_only = _find_dataset_file(dataset_name, "test_image_only")
    if image_only:
        files["test_image_only"] = image_only
    
    table_only = _find_dataset_file(dataset_name, "test_table_only")
    if table_only:
        files["test_table_only"] = table_only
    
    return files


def scan_available_datasets() -> list[str]:
    """
    扫描 datasets/ 目录下可用的数据集
    """
    if not DATASETS_DIR.exists():
        return []
    
    datasets = set()
    
    # 方式1: 查找子目录
    for item in DATASETS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if (item / "train.jsonl").exists():
                datasets.add(item.name)
    
    # 方式2: 查找扁平文件
    for file in DATASETS_DIR.glob("*_train.jsonl"):
        dataset_name = file.stem.replace("_train", "")
        datasets.add(dataset_name)
    
    return sorted(list(datasets))


def get_dataset_config(dataset_name: str, output_base_dir: Optional[str] = None) -> DatasetConfig:
    """
    根据数据集名称获取配置
    """
    dataset_name = dataset_name.lower().strip()
    files = _get_dataset_files(dataset_name)
    
    if "train" not in files:
        available = scan_available_datasets()
        raise ValueError(
            f"找不到数据集 '{dataset_name}' 的训练文件。\n"
            f"请确保以下文件之一存在:\n"
            f"  - {DATASETS_DIR}/{dataset_name}/train.jsonl\n"
            f"  - {DATASETS_DIR}/{dataset_name}_train.jsonl\n"
            f"当前可用的数据集: {available if available else '无'}"
        )
    
    # 输出目录
    output_dir = output_base_dir or os.path.join(DEFAULT_OUTPUT_BASE, dataset_name)
    
    config: DatasetConfig = {
        "train": files.get("train", ""),
        "valid": files.get("valid", files.get("test", "")),
        "test": files.get("test", ""),
        "test_image_only": files.get("test_image_only", ""),
        "test_table_only": files.get("test_table_only", ""),
        "output_base_dir": output_dir,
        "description": f"用户数据集: {dataset_name}",
    }
    
    return config


def list_available_datasets() -> dict[str, str]:
    """列出所有可用的数据集"""
    datasets = scan_available_datasets()
    result = {}
    for name in datasets:
        files = _get_dataset_files(name)
        file_count = len([v for v in files.values() if v])
        result[name] = f"用户数据集 ({file_count}个文件)"
    return result


def generate_dataset_info_entry(dataset_name: str) -> dict:
    """为 LLaMA-Factory 生成 dataset_info.json 格式的配置条目"""
    return {
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "image"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
