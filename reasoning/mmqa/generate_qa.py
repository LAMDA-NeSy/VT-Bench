import json
import os
import sys
import glob
from tqdm import tqdm

# ================= CONFIGURATION =================
CONFIG = {
    # 1. Base directory for the MMQA dataset
    "dataset_dir": "/mnt/hdd/jiazy/MMQA/dataset",
    # 2. Directory where all actual image files are stored
    "image_base_dir": "/mnt/hdd/jiazy/MMQA/final_dataset_images",
    # 3. Final output file path
    "output_filepath": "/mnt/hdd/jiazy/MMQA/mmqa_for_qwen_vl_final.jsonl",
    # 4. Question files to be processed (add "MMQA_test.jsonl" if needed)
    "question_files": ["MMQA_train.jsonl", "MMQA_dev.jsonl", "MMQA_test.jsonl"],
    # 5. Model instruction (forces the model to provide concise answers)
    "answer_instruction": "\n\n--- ANSWER INSTRUCTIONS ---\nProvide *only* the final answer to the question, with no explanation, reasoning, or conversational text. For example, if the answer is 'Mask', just output 'Mask'."
}
# =================================================

def load_knowledge_base(filepath, key_name='id'):
    """
    Loads a JSONL knowledge base file into a dictionary.

    Args:
        filepath (str): Path to the JSONL file.
        key_name (str): The field to use as the dictionary key.

    Returns:
        dict: A dictionary mapping key_name to the full JSON object.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Knowledge base file {filepath} not found. Skipping.")
        return {}
    
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                data[item[key_name]] = item
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping line {i+1} in {filepath} due to error: {e}")
    return data

def linearize_table_to_markdown(table_data):
    """
    Converts a MMQA table JSON object into a Markdown string.

    Args:
        table_data (dict): The table entry from the knowledge base.

    Returns:
        str: A Markdown representation of the table.
    """
    if not table_data or 'table' not in table_data:
        return ""
    
    table_obj = table_data['table']
    header_list = table_obj.get('header', [])
    if not header_list:
        return ""
    
    header_names = [h.get('column_name', '').strip() for h in header_list]
    md_parts = [f"### Table: {table_obj.get('table_name', 'Unnamed Table')}"]
    
    # Build header row
    md_parts.append(f"| {' | '.join(header_names)} |")
    # Build separator row
    md_parts.append(f"| {' | '.join(['---'] * len(header_names))} |")

    # Build data rows
    for row in table_obj.get('table_rows', []):
        cell_texts = [
            cell.get('text', '').strip().replace('\n', ' ').replace('|', '‚') 
            for cell in row
        ]
        if len(cell_texts) == len(header_names):
            md_parts.append(f"| {' | '.join(cell_texts)} |")
            
    return "\n".join(md_parts)

def build_qwen_vl_prompt(question_item, text_kb, table_kb, image_kb, image_base_dir):
    """
    Main logic for associating context and building the Qwen-VL message format.
    """
    metadata = question_item.get("metadata", {})
    text_doc_ids = metadata.get("text_doc_ids", [])
    table_id = metadata.get("table_id")
    image_doc_ids = metadata.get("image_doc_ids", [])

    prompt_content = []

    # 1. Process Images (using glob to match any extension)
    for img_id in image_doc_ids:
        search_pattern = os.path.join(image_base_dir, f"{img_id}.*")
        found_files = glob.glob(search_pattern)
        if found_files:
            # Add absolute path to the prompt
            prompt_content.append({
                "type": "image",
                "image_url": os.path.abspath(found_files[0])
            })
        else:
            print(f"Warning (QID: {question_item['qid']}): Image file for {img_id} not found.", file=sys.stderr)

    # 2. Assemble Textual Context
    text_parts = ["You are an expert multi-modal AI assistant. Please answer the following question based *only* on the provided context."]
    
    # Append Text Snippets
    related_texts = [text_kb.get(tid) for tid in text_doc_ids if text_kb.get(tid)]
    if related_texts:
        formatted_texts = [
            f"--- Text Snippet {i+1} (Title: {t.get('title', 'N/A')}) ---\n{t['text']}" 
            for i, t in enumerate(related_texts)
        ]
        text_parts.append("--- START TEXT CONTEXT ---\n\n" + "\n\n".join(formatted_texts) + "\n--- END TEXT CONTEXT ---")

    # Append Table
    related_table = table_kb.get(table_id)
    if related_table:
        table_md = linearize_table_to_markdown(related_table)
        if table_md:
            text_parts.append("--- START TABLE CONTEXT ---\n\n" + table_md + "\n--- END TABLE CONTEXT ---")

    # 3. Add Final Question + Answer Instructions (merged from code step 3)
    text_parts.append(f"--- QUESTION ---\n{question_item['question']}{CONFIG['answer_instruction']}")
    
    # Finalize text block
    prompt_content.append({
        "type": "text",
        "text": "\n\n".join(text_parts)
    })

    return [{"role": "user", "content": prompt_content}]

def main():
    """
    Main execution pipeline: Load KBs -> Merge Data -> Build Prompts -> Save.
    """
    # 1. Load Knowledge Bases into memory
    print("Loading Knowledge Bases...")
    text_kb = load_knowledge_base(os.path.join(CONFIG["dataset_dir"], "MMQA_texts.jsonl"))
    table_kb = load_knowledge_base(os.path.join(CONFIG["dataset_dir"], "MMQA_tables.jsonl"))
    image_kb = load_knowledge_base(os.path.join(CONFIG["dataset_dir"], "MMQA_images.jsonl"))
    print(f"KBs loaded: Texts({len(text_kb)}), Tables({len(table_kb)}), Images({len(image_kb)})")

    # 2. Process Questions and Generate Inference Data
    final_output = []
    for q_filename in CONFIG["question_files"]:
        q_path = os.path.join(CONFIG["dataset_dir"], q_filename)
        if not os.path.exists(q_path):
            print(f"Skipping {q_filename}: File not found.")
            continue
        
        print(f"Processing question file: {q_filename}")
        with open(q_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Building Prompts for {q_filename}"):
                try:
                    question_item = json.loads(line)
                    
                    # Core logic: Merge context and build Qwen-VL prompt structure
                    model_input = build_qwen_vl_prompt(
                        question_item, text_kb, table_kb, image_kb, CONFIG["image_base_dir"]
                    )
                    
                    # Keep metadata and ground truth for evaluation later
                    final_output.append({
                        "qid": question_item["qid"],
                        "question": question_item["question"],
                        "answers": question_item["answers"],
                        "metadata": question_item.get("metadata", {}),
                        "model_input": model_input
                    })
                except Exception as e:
                    print(f"Error processing item: {e}")

    # 3. Write final output to disk
    print(f"Saving final dataset to: {CONFIG['output_filepath']}")
    with open(CONFIG['output_filepath'], 'w', encoding='utf-8') as f:
        for item in final_output:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Success! Processed {len(final_output)} total samples.")
    print("The file is now ready for Qwen-VL inference.")

if __name__ == "__main__":
    main()