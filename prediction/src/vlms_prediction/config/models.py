# 模型配置映射
# 将简短的模型名称映射到完整的模型配置

from typing import TypedDict, Optional, Union, List


class ModelConfig(TypedDict):
    """模型配置类型"""
    model_name_or_path: str      # HuggingFace 模型路径或本地路径
    template: str                # 对话模板类型
    image_max_pixels: int        # 图像最大像素数
    video_max_pixels: Optional[int]  # 视频最大像素数（可选）
    trust_remote_code: bool      # 是否信任远程代码
    lora_rank: int               # LoRA rank
    lora_alpha: int              # LoRA alpha
    lora_dropout: float          # LoRA dropout
    lora_target: Union[str, List[str]]  # LoRA 目标层
    cutoff_len: int              # 序列最大长度
    description: str             # 模型描述


# 模型配置映射表（仅保留 Table-LLaVA 和 Qwen3-VL）
MODEL_CONFIG: dict[str, ModelConfig] = {
    # Table-LLaVA-7B（专为表格数据优化）
    "tablellava-7b": {
        "model_name_or_path": "SpursgoZmy/table-llava-v1.5-7b-hf",
        "template": "llava",
        "image_max_pixels": 112896,  # 336x336
        "video_max_pixels": None,
        "trust_remote_code": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": ["q_proj", "v_proj"],
        "cutoff_len": 2048,
        "description": "Table-LLaVA 7B - 专为表格理解优化的视觉语言模型",
    },
    
    # Qwen3-VL-8B-Instruct（推荐用于大多数任务）
    "qwen3-vl-8b-instruct": {
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        "template": "qwen2_vl",
        "image_max_pixels": 1048576,  # 1024x1024
        "video_max_pixels": 16384,
        "trust_remote_code": True,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "cutoff_len": 4096,
        "description": "Qwen3 视觉语言模型 8B 参数版本 - 性能最强",
    },
}

# 模型别名映射（支持多种输入方式）
MODEL_ALIASES: dict[str, str] = {
    # Table-LLaVA 别名
    "tablellava-7b": "tablellava-7b",
    "tablellava": "tablellava-7b",
    "table-llava": "tablellava-7b",
    "table_llava": "tablellava-7b",
    "table-llava-v1.5-7b-hf": "tablellava-7b",
    "spursgozmy/table-llava-v1.5-7b-hf": "tablellava-7b",
    
    # Qwen3-VL 别名
    "qwen3-vl-8b-instruct": "qwen3-vl-8b-instruct",
    "qwen3-vl-8b": "qwen3-vl-8b-instruct",
    "qwen3-vl": "qwen3-vl-8b-instruct",
    "qwen3": "qwen3-vl-8b-instruct",
    "qwen": "qwen3-vl-8b-instruct",
    "qwen/qwen3-vl-8b-instruct": "qwen3-vl-8b-instruct",
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    根据模型名称获取配置
    
    Args:
        model_name: 模型名称（支持别名）
        
    Returns:
        ModelConfig: 模型配置
        
    Raises:
        ValueError: 如果模型名称不存在
    """
    # 转换为小写并尝试匹配别名
    model_name_lower = model_name.lower().strip().replace("_", "-")
    
    if model_name_lower in MODEL_ALIASES:
        canonical_name = MODEL_ALIASES[model_name_lower]
        return MODEL_CONFIG[canonical_name]
    
    # 尝试直接匹配
    if model_name_lower in MODEL_CONFIG:
        return MODEL_CONFIG[model_name_lower]
    
    # 列出所有可用的模型
    available = list(MODEL_CONFIG.keys())
    raise ValueError(
        f"未知的模型: '{model_name}'\n"
        f"可用的模型: {available}"
    )


def list_available_models() -> dict[str, str]:
    """列出所有可用的模型及其描述"""
    return {name: config["description"] for name, config in MODEL_CONFIG.items()}
