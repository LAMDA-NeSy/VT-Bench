import sys
import argparse
from pathlib import Path

# 1. 核心路径修正
BASE_PATH = Path(__file__).resolve().parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

def evaluate_dvm(model_id: str, tasks: list = None, output_dir: str = "."):
    model_id_lower = model_id.lower()
    is_proprietary = any(brand in model_id_lower for brand in ["gpt", "gemini", "o1"])

    if is_proprietary:
        # 2. 只有在调用时才导入，提高灵活性
        from .test_proprietary import run_dvm_benchmark_api
        return run_dvm_benchmark_api(model_id, tasks=tasks, output_root=output_dir)
    else:
        from .test_open_model import evaluate_dvm_benchmark
        return evaluate_dvm_benchmark(model_id, task_types=tasks, output_dir=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVM Multimodal Benchmark Router")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID or Path")
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to evaluate: loc attr count mean")
    parser.add_argument("--output", type=str, default=".", help="Root directory for output")
    
    args = parser.parse_args()

    # 执行评测
    evaluate_dvm(model_id=args.model_id, tasks=args.tasks, output_dir=args.output)