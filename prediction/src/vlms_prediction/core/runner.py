# VLMs Prediction 核心运行器
# 提供统一的 VLMs_prediction() 函数接口

import os
import sys
import json
import yaml
import subprocess
from typing import Any, Literal, Optional
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..config.datasets import (
    get_dataset_config, 
    list_available_datasets, 
    scan_available_datasets,
    generate_dataset_info_entry,
    DATASETS_DIR,
    DatasetConfig
)
from ..config.models import get_model_config, list_available_models, ModelConfig
from ..config.base_config import get_base_training_config, get_preset_config


# 诊断模式类型
DiagnosisMode = Literal["full", "mcr"]

# LLaMA-Factory 项目根目录
LLAMAFACTORY_ROOT = Path(__file__).parent.parent.parent.parent


def _setup_environment():
    """
    设置环境变量（使用用户友好的默认值）
    可通过环境变量覆盖：
      - HF_HOME: HuggingFace 缓存目录
      - TMPDIR: 临时文件目录
    """
    # 默认使用用户 home 目录下的 .cache
    default_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    default_tmp = os.path.join(os.path.expanduser("~"), "tmp")
    
    os.environ.setdefault("HF_HOME", default_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", default_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", default_cache)
    os.environ.setdefault("TMPDIR", default_tmp)
    os.environ.setdefault("TMP", default_tmp)


def _create_dataset_info(dataset_config: DatasetConfig, dataset_name: str) -> str:
    """
    创建临时的 dataset_info.json 文件供 LLaMA-Factory 使用
    """
    output_dir = dataset_config["output_base_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_info = {}
    base_entry = generate_dataset_info_entry(dataset_name)
    
    if dataset_config["train"]:
        dataset_info[f"{dataset_name}_train"] = {
            **base_entry,
            "file_name": dataset_config["train"]
        }
    
    if dataset_config["valid"]:
        dataset_info[f"{dataset_name}_valid"] = {
            **base_entry,
            "file_name": dataset_config["valid"]
        }
    
    if dataset_config["test"]:
        dataset_info[f"{dataset_name}_test"] = {
            **base_entry,
            "file_name": dataset_config["test"]
        }
    
    if dataset_config["test_image_only"]:
        dataset_info[f"{dataset_name}_image_only_test"] = {
            **base_entry,
            "file_name": dataset_config["test_image_only"]
        }
    
    if dataset_config["test_table_only"]:
        dataset_info[f"{dataset_name}_tabular_only_test"] = {
            **base_entry,
            "file_name": dataset_config["test_table_only"]
        }
    
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    return info_path


def _build_training_args(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    dataset_name: str,
    diagnosis: str,
    dataset_info_path: str,
    **kwargs
) -> dict[str, Any]:
    """
    构建训练参数字典
    """
    base_config = get_base_training_config()
    
    args = {
        **base_config,
        
        # 模型配置
        "model_name_or_path": model_config["model_name_or_path"],
        "template": model_config["template"],
        "image_max_pixels": model_config["image_max_pixels"],
        "trust_remote_code": model_config["trust_remote_code"],
        "cutoff_len": model_config["cutoff_len"],
        
        # LoRA 配置
        "lora_rank": model_config["lora_rank"],
        "lora_alpha": model_config["lora_alpha"],
        "lora_dropout": model_config["lora_dropout"],
        "lora_target": model_config["lora_target"],
        
        # 数据集配置
        "dataset": f"{dataset_name}_train",
        "eval_dataset": f"{dataset_name}_valid" if dataset_config["valid"] else None,
        "dataset_dir": os.path.dirname(dataset_info_path),
    }
    
    if model_config["video_max_pixels"]:
        args["video_max_pixels"] = model_config["video_max_pixels"]
    
    # 输出目录
    model_short_name = model_config["model_name_or_path"].split("/")[-1].lower()
    output_dir = os.path.join(
        dataset_config["output_base_dir"],
        f"{model_short_name}_{diagnosis}"
    )
    args["output_dir"] = output_dir
    
    # 移除 None 值
    args = {k: v for k, v in args.items() if v is not None}
    
    args.update(kwargs)
    
    return args


def _run_single_training(args: dict[str, Any], dry_run: bool = False) -> Optional[str]:
    """
    运行单次训练
    """
    if dry_run:
        print("=" * 60)
        print("训练配置（dry_run 模式，不会实际运行）:")
        print("=" * 60)
        for key, value in sorted(args.items()):
            print(f"  {key}: {value}")
        print("=" * 60)
        return args.get("output_dir")
    
    output_dir = args.get("output_dir", "/tmp/vlms_output")
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "train_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(args, f, default_flow_style=False, allow_unicode=True)
    
    print("=" * 60)
    print(f"开始训练...")
    print(f"数据集: {args['dataset']}")
    print(f"模型: {args['model_name_or_path']}")
    print(f"输出目录: {output_dir}")
    print(f"配置文件: {config_path}")
    print("=" * 60)
    
    cmd = ["llamafactory-cli", "train", config_path]
    print(f"\n运行命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=str(LLAMAFACTORY_ROOT))
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"\n训练失败，退出码: {e.returncode}")
        raise RuntimeError(f"训练失败: {e}")


def VLMs_prediction(
    data: str,
    model: str,
    diagnosis: DiagnosisMode = "full",
    output_dir: Optional[str] = None,
    # 训练参数覆盖
    num_epochs: Optional[float] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_samples: Optional[int] = None,
    # 预设配置
    preset: Optional[str] = None,
    # 运行控制
    dry_run: bool = False,
    verbose: bool = True,
    **kwargs
) -> dict[str, Any]:
    """
    VLMs 预测主函数 - 运行视觉语言模型的微调任务
    
    Args:
        data: 数据集名称（对应 datasets/ 目录下的子目录名或文件前缀）
        model: 模型名称 ("qwen3", "tablellava")
        diagnosis: 诊断模式 ("full", "mcr")
        output_dir: 自定义输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        max_samples: 最大样本数
        preset: 预设配置
        dry_run: 是否只打印配置
        verbose: 是否打印详细信息
        
    Returns:
        运行结果字典
    """
    _setup_environment()
    
    result = {
        "status": "success",
        "experiments": [],
        "output_dirs": [],
        "config_files": [],
        "message": "",
    }
    
    try:
        dataset_config = get_dataset_config(data, output_dir)
        model_config = get_model_config(model)
        
        if verbose:
            print("=" * 60)
            print("VLMs Prediction 任务配置")
            print("=" * 60)
            print(f"数据集: {data}")
            print(f"模型: {model} - {model_config['description']}")
            print(f"诊断模式: {diagnosis}")
            print(f"数据集目录: {DATASETS_DIR}")
            print(f"输出目录: {dataset_config['output_base_dir']}")
            print("=" * 60)
        
        dataset_info_path = _create_dataset_info(dataset_config, data)
        if verbose:
            print(f"生成 dataset_info.json: {dataset_info_path}")
        
        extra_args = {}
        
        if preset:
            preset_config = get_preset_config(preset)
            extra_args.update(preset_config)
            if verbose:
                print(f"应用预设配置: {preset}")
        
        if num_epochs is not None:
            extra_args["num_train_epochs"] = num_epochs
        if batch_size is not None:
            extra_args["per_device_train_batch_size"] = batch_size
        if learning_rate is not None:
            extra_args["learning_rate"] = learning_rate
        if max_samples is not None:
            extra_args["max_samples"] = max_samples
        
        extra_args.update(kwargs)
        
        if diagnosis == "full":
            args = _build_training_args(
                dataset_config, model_config, data, "full", dataset_info_path, **extra_args
            )
            out_dir = _run_single_training(args, dry_run)
            result["experiments"].append("full")
            result["output_dirs"].append(out_dir)
            
        elif diagnosis == "mcr":
            if verbose:
                print("\n" + "=" * 60)
                print("MCR 消融实验")
                print("  1. image_only - 仅图像模态")
                print("  2. table_only - 仅表格模态")
                print("=" * 60 + "\n")
            
            if not dataset_config["test_image_only"]:
                raise ValueError(f"消融实验需要 {data}/test_image_only.jsonl 文件")
            if not dataset_config["test_table_only"]:
                raise ValueError(f"消融实验需要 {data}/test_table_only.jsonl 文件")
            
            # image_only
            args_image = _build_training_args(
                dataset_config, model_config, data, "image_only", dataset_info_path, **extra_args
            )
            args_image["eval_dataset"] = f"{data}_image_only_test"
            out_dir_image = _run_single_training(args_image, dry_run)
            result["experiments"].append("image_only")
            result["output_dirs"].append(out_dir_image)
            
            # table_only
            args_table = _build_training_args(
                dataset_config, model_config, data, "table_only", dataset_info_path, **extra_args
            )
            args_table["eval_dataset"] = f"{data}_tabular_only_test"
            out_dir_table = _run_single_training(args_table, dry_run)
            result["experiments"].append("table_only")
            result["output_dirs"].append(out_dir_table)
        
        else:
            raise ValueError(f"未知的诊断模式: {diagnosis}，可用: full, mcr")
        
        result["message"] = f"成功完成 {len(result['experiments'])} 个实验"
        
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        if verbose:
            print(f"\n错误: {e}")
        raise
    
    if verbose:
        print("\n" + "=" * 60)
        print("任务完成!")
        print(f"状态: {result['status']}")
        print(f"实验: {result['experiments']}")
        print(f"输出目录: {result['output_dirs']}")
        print("=" * 60)
    
    return result


# 便捷函数
def list_datasets() -> dict[str, str]:
    return list_available_datasets()

def list_models() -> dict[str, str]:
    return list_available_models()

def show_dataset_dir() -> str:
    return str(DATASETS_DIR)


# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLMs Prediction - 视觉语言模型多模态微调")
    parser.add_argument("--data", "-d", help="数据集名称")
    parser.add_argument("--model", "-m", help="模型名称 (tablellava, qwen3)")
    parser.add_argument("--diagnosis", "-D", default="full", choices=["full", "mcr"])
    parser.add_argument("--output-dir", "-o", help="输出目录")
    parser.add_argument("--preset", "-p", help="预设配置")
    parser.add_argument("--epochs", type=float, help="训练轮数")
    parser.add_argument("--batch-size", type=int, help="批次大小")
    parser.add_argument("--learning-rate", type=float, help="学习率")
    parser.add_argument("--max-samples", type=int, help="最大样本数")
    parser.add_argument("--dry-run", action="store_true", help="只打印配置")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--show-dataset-dir", action="store_true")
    
    args = parser.parse_args()
    
    if args.show_dataset_dir:
        print(f"\n数据集目录: {show_dataset_dir()}")
        sys.exit(0)
    
    if args.list_datasets:
        datasets = list_datasets()
        print(f"\n数据集目录: {show_dataset_dir()}")
        print("\n可用的数据集:")
        if datasets:
            for name, desc in datasets.items():
                print(f"  - {name}")
        else:
            print("  (无数据集)")
        sys.exit(0)
        
    if args.list_models:
        print("\n可用的模型:")
        for name, desc in list_models().items():
            print(f"  - {name}: {desc}")
        sys.exit(0)
    
    if not args.data or not args.model:
        parser.print_help()
        sys.exit(1)
    
    VLMs_prediction(
        data=args.data,
        model=args.model,
        diagnosis=args.diagnosis,
        output_dir=args.output_dir,
        preset=args.preset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )
