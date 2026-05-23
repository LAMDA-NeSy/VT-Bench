# VLMs Prediction Module
# 用于视觉语言模型的多模态预测任务
# 
# 使用方法:
#   from vlms_prediction import VLMs_prediction
#   VLMs_prediction(data="breast", model="qwen3-vl-8b-instruct", diagnosis="full")

from .core.runner import VLMs_prediction

__all__ = ["VLMs_prediction"]
__version__ = "1.0.0"

