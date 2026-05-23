import json
import os
import glob

def verify_reasoning_consistency(file_path):
    """
    检测模型输出中是否准确识别并提到了决策所需的视觉属性值。
    适配 DVM Car QA 结果文件。
    """
    stats = {
        "total": 0,
        "correct_recognition": 0,
        "failed_recognition": 0,
        "details": []
    }
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # 从文件名推断模式 (用于兼容旧逻辑)
    file_name_lower = os.path.basename(file_path).lower()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
            except:
                continue
            
            # 1. 获取元数据和内容
            # 优先从 item["id"] 或文件名判断模式
            task_id = item.get("id", "").lower()
            raw_content = item.get("raw_output", "").lower()
            image_rel = item.get("image_rel", "")
            
            # 2. 从路径提取 Ground Truth 属性值
            # 结构: resized_DVM/Maker/Genmodel/Reg_year/Color/...
            parts = image_rel.split('/')
            if len(parts) < 6:
                continue
            
            gt_maker = parts[1].lower()
            gt_genmodel = parts[2].lower() # 例如 Yaris
            gt_color = parts[4].lower()
            
            # 3. 确定检测模式 (Logic Detection)
            # 如果 id 包含 maker_color 或文件名包含它，则检测两项
            check_maker = "maker" in task_id or "maker" in file_name_lower
            check_color = "color" in task_id or "color" in file_name_lower
            
            # 4. 执行检测逻辑
            recognized = True
            missing = []

            if check_maker:
                # 检查品牌 (Toyota) 或 具体车型 (Yaris)
                if gt_maker not in raw_content and gt_genmodel not in raw_content:
                    recognized = False
                    missing.append(f"maker:{gt_maker}")
            
            if check_color:
                if gt_color not in raw_content:
                    recognized = False
                    missing.append(f"color:{gt_color}")

            # 5. 记录结果
            stats["total"] += 1
            if recognized:
                stats["correct_recognition"] += 1
            else:
                stats["failed_recognition"] += 1
                stats["details"].append({
                    "id": item.get("id"),
                    "expected_missing": missing,
                    "raw_output_snippet": raw_content[:100].replace("\n", " ") # 截取部分输出方便调试
                })
    
    return stats

# === 使用示例 ===
# 这里保持你的路径不变
output_files = glob.glob("/mnt/hdd/jiazy/DVM_Car_QA/stage1/thinking_result/ATTR/dvm_*_attr_*rows.results.jsonl")

for f_path in output_files:
    report = verify_reasoning_consistency(f_path)
    if report and report["total"] > 0:
        accuracy = (report["correct_recognition"] / report["total"]) * 100
        print(f"File: {os.path.basename(f_path)}")
        print(f"  - Total Samples: {report['total']}")
        print(f"  - Reasoning Recognition Accuracy: {accuracy:.2f}%")
        if report["failed_recognition"] > 0:
            # 打印前2个失败案例，方便观察模型到底哪里漏说了
            print(f"  - Sample Failures: {report['details'][:2]}")
        print("-" * 50)