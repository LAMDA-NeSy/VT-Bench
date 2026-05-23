import json
import os
import sys
import string
import re
from tqdm import tqdm
from collections import defaultdict
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from PIL import Image

# -----------------------------------------------------------
# 1. Text Normalization & Evaluation Utilities
# -----------------------------------------------------------

def normalize_text(s):
    """Standard EM normalization: lowercase, remove punctuation and articles."""
    s = str(s).lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_prediction_text(pred_text):
    """
    Cleans model output by removing common prefixes and taking the first coherent sentence.
    """
    s = pred_text.strip()
    prefixes_to_split = [
        "the final answer is:", "the answer is:", "my answer is:",
        "the answer is", "is:", "was:", "are:", "were:"
    ]
    s_lower = s.lower()
    last_prefix_pos, prefix_len = -1, 0
    
    for prefix in prefixes_to_split:
        pos = s_lower.rfind(prefix)
        if pos > last_prefix_pos:
            last_prefix_pos, prefix_len = pos, len(prefix)
            
    if last_prefix_pos != -1:
        s = s[last_prefix_pos + prefix_len:].strip()
        
    # Take the first line and truncate at first end-of-sentence punctuation
    s = s.split('\n')[0].strip()
    s = re.split(r'[\.\!\?]\s+', s, maxsplit=1)[0].strip()
    
    # Remove wrapping quotes
    if s.startswith('"') and s.endswith('"') and len(s) > 2:
        s = s[1:-1]
    return s.strip()

# -----------------------------------------------------------
# 2. Llama 3.2 Vision Inference Engine
# -----------------------------------------------------------

class Llama3VisionEngine:
    def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        print(f"--- Loading Llama 3.2 Vision: {model_id} ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # Handle pad_token_id fallback
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        
        print(f"Model loaded successfully on {self.model.device}")

    def _prepare_inputs(self, raw_messages):
        """Processes raw MMQA messages into Llama-compatible format."""
        processed_msgs = []
        images = []

        for msg in raw_messages:
            role = msg["role"]
            content_list = msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
            
            new_contents = []
            for item in content_list:
                if item["type"] == "image":
                    try:
                        img = Image.open(item["image_url"]).convert("RGB")
                        images.append(img)
                        # Placeholder for Llama-3.2 Vision
                        new_contents.append({"type": "image"}) 
                    except Exception as e:
                        print(f"[WARN] Failed to load image {item['image_url']}: {e}")
                elif item["type"] == "text":
                    new_contents.append({"type": "text", "text": item["text"]})
            
            processed_msgs.append({"role": role, "content": new_contents})

        # Apply chat template to get the final prompt string
        input_text = self.processor.apply_chat_template(
            processed_msgs,
            add_generation_prompt=True,
            tokenize=False 
        )

        # Tokenize and create image tensors
        inputs = self.processor(
            images=images if images else None,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
        
        return inputs

    @torch.no_grad()
    def predict(self, raw_messages, max_new_tokens=1024):
        """Main inference method returning the model's textual response."""
        inputs = self._prepare_inputs(raw_messages)
        
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id
        )

        # Slice to get only the newly generated response tokens
        input_len = inputs["input_ids"].shape[1]
        response_ids = gen_ids[0, input_len:]
        response_text = self.processor.decode(response_ids, skip_special_tokens=True).strip()
        
        return response_text

# -----------------------------------------------------------
# 3. Main Evaluation Interface
# -----------------------------------------------------------

def run_llama_evaluation(input_file="mmqa_for_qwen_vl_inference_patched.jsonl",
        output_file="result/llama3_2_results.jsonl", 
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    """
    Core function to run MMQA evaluation for Llama 3.2 Vision.
    """
    Image.MAX_IMAGE_PIXELS = None
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Initialize Engine
    engine = Llama3VisionEngine(model_id)
    
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct, overall_total = 0, 0
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Evaluating Llama 3.2"):
            try:
                sample = json.loads(line)
                qid = sample.get('qid', 'unknown')
                q_type = sample.get('metadata', {}).get('type', 'Unknown')
                ground_truths = [ans['answer'] for ans in sample.get('answers', [])]
                
                # Perform Inference
                raw_response = engine.predict(sample['model_input'])
                
                # Evaluation Logic
                cleaned_response = clean_prediction_text(raw_response)
                pred_norm = normalize_text(cleaned_response)
                gt_norms = [normalize_text(gt) for gt in ground_truths]
                
                is_correct = (pred_norm in gt_norms) if gt_norms else False

                # Statistics
                overall_total += 1
                type_stats[q_type]["total"] += 1
                if is_correct:
                    overall_correct += 1
                    type_stats[q_type]["correct"] += 1

                # Save Result Entry
                result = {
                    "qid": qid,
                    "question_type": q_type,
                    "prediction_raw": raw_response,
                    "prediction_cleaned": cleaned_response,
                    "ground_truths": ground_truths,
                    "is_correct": is_correct
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush() # Ensure data is saved incrementally

            except Exception as e:
                print(f"\n[ERROR] Exception on QID {sample.get('qid', 'N/A')}: {e}")
                continue

    _print_summary(type_stats, overall_correct, overall_total)

def _print_summary(type_stats, correct, total):
    print("\n" + "="*60)
    print(f"{'Question Type':<40} | {'Accuracy'}")
    print("-" * 60)
    for q_type in sorted(type_stats.keys()):
        stats = type_stats[q_type]
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"{q_type[:40]:<40} | {acc:6.2f}% ({stats['correct']}/{stats['total']})")

    overall_acc = (correct / total * 100) if total > 0 else 0
    print("-" * 60)
    print(f"{'OVERALL ACCURACY (EM)':<40} | {overall_acc:6.2f}% ({correct}/{total})")
    print("="*60)

if __name__ == "__main__":
    run_llama_evaluation(
        input_file="mmqa_for_qwen_vl_inference_patched.jsonl",
        output_file="result/llama3_2_results.jsonl"
    )