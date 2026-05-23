# 基础训练配置
# 定义默认的训练参数，可根据需要覆盖

from typing import Any, Optional


def get_base_training_config(
    num_epochs: float = 3.0,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,
    eval_steps: int = 500,
    save_steps: int = 500,
    logging_steps: int = 50,
    use_bf16: bool = True,
    use_flash_attn: bool = True,
    quantization_bit: Optional[int] = None,
) -> dict[str, Any]:
    """
    获取基础训练配置
    
    Args:
        num_epochs: 训练轮数
        batch_size: 每设备批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        warmup_ratio: 预热比例
        eval_steps: 评估间隔步数
        save_steps: 保存间隔步数
        logging_steps: 日志记录间隔步数
        use_bf16: 是否使用 bf16 精度
        use_flash_attn: 是否使用 flash attention
        quantization_bit: 量化位数（None表示不量化，4或8表示量化）
        
    Returns:
        训练配置字典
    """
    config = {
        # 训练阶段
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        
        # 训练参数
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": warmup_ratio,
        
        # 评估参数
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        
        # 保存和日志
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "none",
        
        # 优化器
        "optim": "adamw_torch",
        "upcast_layernorm": True,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        
        # 精度和加速
        "bf16": use_bf16,
        "flash_attn": "auto" if use_flash_attn else "disabled",
        
        # 数据处理
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        "dataloader_num_workers": 4,
        
        # 分布式训练超时
        "ddp_timeout": 180000000,
        
        # 断点续训
        "resume_from_checkpoint": None,
    }
    
    # 量化配置
    if quantization_bit is not None:
        config["quantization_bit"] = quantization_bit
        config["optim"] = "paged_adamw_8bit"  # 量化时使用 8bit 优化器
    
    return config


# 预设配置：针对不同场景优化
PRESET_CONFIGS = {
    # 快速测试配置
    "debug": {
        "num_epochs": 1.0,
        "max_samples": 100,
        "eval_steps": 50,
        "save_steps": 50,
        "logging_steps": 10,
    },
    
    # 小数据集配置（< 5000 样本）
    "small_dataset": {
        "num_epochs": 5.0,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.1,
        "eval_steps": 100,
        "save_steps": 100,
    },
    
    # 中等数据集配置（5000-30000 样本）
    "medium_dataset": {
        "num_epochs": 3.0,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.05,
        "eval_steps": 500,
        "save_steps": 500,
    },
    
    # 大数据集配置（> 30000 样本）
    "large_dataset": {
        "num_epochs": 2.0,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.03,
        "eval_steps": 1000,
        "save_steps": 1000,
    },
    
    # 低显存配置
    "low_memory": {
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "quantization_bit": 4,
    },
}


def get_preset_config(preset_name: str) -> dict[str, Any]:
    """
    获取预设配置
    
    Args:
        preset_name: 预设名称（debug, small_dataset, medium_dataset, large_dataset, low_memory）
        
    Returns:
        预设配置字典
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"未知的预设: {preset_name}, 可用: {list(PRESET_CONFIGS.keys())}")
    
    return PRESET_CONFIGS[preset_name]

