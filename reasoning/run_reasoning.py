import os
import sys

# Importing dataset-specific interfaces
from .DVM_QA.test_model import evaluate_dvm
from .ehrxqa.test_model import ehrxqa_benchmark_interface
from .mmqa.test_models import evaluate_model

def reasoning_evaluation(model_id: str, dataset: str, setting: str):
    """
    Central dispatcher to route evaluation based on the chosen dataset.
    
    Args:
        model_id (str): The identifier of the model to evaluate.
        dataset (str): The target dataset name ('dvm', 'ehrxqa', etc.).
        setting (str): The evaluation mode ('stage1', 'stage2', or 'full').
    """
    dataset_type = dataset.lower().strip()
    
    print(f"[*] Global Dispatcher | Dataset: {dataset_type} | Model: {model_id} | Setting: {setting}")

    # 1. Route to DVM Dataset Logic
    if dataset_type == "dvm":
        print(f"[INFO] Routing to DVM Evaluation...")
        return evaluate_dvm(model_id=model_id, setting=setting)

    # 2. Route to EHRXQA Dataset Logic
    elif dataset_type == "ehrxqa":
        print(f"[INFO] Routing to EHRXQA Benchmark Interface...")
        return ehrxqa_benchmark_interface(model_id=model_id, setting=setting)

    # 3. Fallback for other datasets (e.g., MultimodalQA / mmqa)
    else:
        print(f"[INFO] Dataset '{dataset}' not specifically routed. Calling generic evaluate_model...")
        return evaluate_model(model_id=model_id)

# Example Usage:
# reasoning_evaluation("gpt-4.1", "ehrxqa", "full")
# reasoning_evaluation("Qwen/Qwen3-VL-8B-Instruct", "dvm", "stage2")
# reasoning_evaluation("internvl2-8b", "mmqa", "stage1")