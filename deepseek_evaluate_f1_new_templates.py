import json
import re
import time
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import matplotlib.pyplot as plt

#prompt_templates
PROMPT_TEMPLATES = {
"zero_shot_technical": """
Please extract at most {max_k} academic keywords strictly from the following text, the keywords：
1. Must be a complete phrase that appears continuously in the original text（e.g. do not split "high Reynolds number"）
2. Exactly matches the example format：
Right："nuclear theory", "quantum liquids"
Wrong："theory", "liquids"
Text：\"\"\"{text}\"\"\"
Keywords list（one per line，no numbers or punctuation）：
""",
"few_shot_standard": """
example：
text：\"\"\"{example1_text}\"\"\"
keywords：{example1_keywords}

text：\"\"\"{example2_text}\"\"\"
keywords：{example2_keywords}

Now please extract keywords from the following text：
text：\"\"\"{text}\"\"\"
"""
}

# Main set
API_KEY = ""  #### Your own DeepSeek API
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

k_list = [5, 15, 25, 35]  
MAX_EXAMPLES = 350 

# In this file, the dataset file is train_instruction,json, which is not cleaned. #######
# So, there are some codes for data cleaning. And the dataset directory should be absolute directory. 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))


def parse_keywords(api_response: str) -> List[str]:

    if not api_response:
        return []


    try:
        import json
        data = json.loads(api_response)
        if isinstance(data, dict) and "keywords" in data:
            return data["keywords"]
    except:
        pass

    markdown_keywords = re.findall(r'\*\*([^*]+)\*\*', api_response)
    if markdown_keywords:
        return [kw.strip() for kw in markdown_keywords][:10] 

    numbered_keywords = re.findall(r'\d+\.\s*([^\n]+)', api_response)
    if numbered_keywords:
        return [kw.strip() for kw in numbered_keywords][:10]

    return [kw.strip() for kw in api_response.split('\n') if kw.strip()][:10]
# call_API function (be modified)
def call_keyword_extraction_api(
    text: str,
    template_name="zero_shot_technical",
    max_k: int = 5
    ) -> List[str]:
    try:
        template = PROMPT_TEMPLATES[template_name].format(
            text=text,max_k=max_k
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": template}],
            temperature=0.3,
            max_tokens=100 
        )
        #print(template)
        # Parse keywords
        raw_output = response.choices[0].message.content
        #print("Response of API:",raw_output)
        keywords = parse_keywords(raw_output)
        print("keywords:",keywords)
        # Verify whether the keywords are in the original text
        valid_keywords = [
            kw for kw in keywords
            if kw.lower() in text.lower()
        ]

        if len(valid_keywords) < len(keywords):
            logger.warning(
                f"Filter out {len(keywords)-len(valid_keywords)} keywords that are not in the original text"
            )

        return valid_keywords[:max_k]

    except Exception as e:
        logger.error(f"Fail to call API: {str(e)}")
        raise
# === Further clean the keywords returned by the API ====
def clean_predicted_keywords(keywords: List[str]) -> List[str]:
    cleaned = []
    for kw in keywords:
        words = kw.split()
        if len(words) <= 3:
            cleaned.append(' '.join(words))
    return cleaned


# F1-score
def evaluate_f1_at_k(predicted_keywords: List[str], true_keywords: List[str], k: int, prompt_info="") -> float:
    
    def preprocess(keyword:str) -> str:
        return re.sub(r'[^\w\s]','',keyword.lower()).strip()
    
    true_processed = list({preprocess(kw) for kw in true_keywords}) # Remove duplicates
    pred_processed = [preprocess(kw) for kw in predicted_keywords[:k]]
    pred_processed = clean_predicted_keywords(pred_processed)
    # Compute True Positives (allow substring matching)
    tp = 0
    matched_true_indices = set()

    for pred_kw in pred_processed:
        for i, true_kw in enumerate(true_processed):
            if i not in matched_true_indices and (pred_kw in true_kw or true_kw in pred_kw):
                tp += 1
                matched_true_indices.add(i)
                break
    #print("\n=== Keywords Comparison ===")
    #print(f"True Keywords: {true_keywords}")
    #print(f"Predicted Keywords: {predicted_keywords}")
    pred_set = set(predicted_keywords[:k])
    gt_set = set(true_keywords)
    # calculate tp, substring matching is allowed
    tp = 0
    matched_true_indices = set()
    for pred_kw in pred_processed:
        for i, true_kw in enumerate(true_processed):
            if i not in matched_true_indices and (pred_kw in true_kw or true_kw in pred_kw):
                tp += 1
                matched_true_indices.add(i)
                break
    fp = len(pred_processed) - tp
    fn = len(true_processed) - tp

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'prompt_informaiton': prompt_info
        }
    #print(metrics)
    return f1
    # return 2 * precision * recall / (precision + recall)


# Main evaluate
"""
def evaluate():
    with open("~/train_instruction.json", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    n = min(len(data), MAX_EXAMPLES)
    total_f1_scores = {k: 0.0 for k in K_LIST}
    results = []

    for i in tqdm(range(n), desc="Evaluating"):
        item = data[i]
        gt_keywords = [kw.strip().lower() for kw in item["output"].split(",")]

        try:
            prediction = generate_keywords(item["input"], item["instruction"])
            pred_keywords = [kw.strip().lower() for kw in prediction.split(",")]

            result = {
                "instruction": item["instruction"],
                "input": item["input"],
                "ground_truth": gt_keywords,
                "prediction_raw": prediction,
                "prediction_list": pred_keywords,
                "f1_scores": {}
            }

            for k in K_LIST:
                f1 = compute_f1_at_k(pred_keywords, gt_keywords, k)
                total_f1_scores[k] += f1
                result["f1_scores"][f"f1@{k}"] = round(f1, 4)

            results.append(result)

        except Exception as e:
            print(f"[ERROR] Skipped example {i}: {e}")
            continue

        time.sleep(0.5)  # 防止API速率限制

    # === 打印最终结果 ===
    print("\n==== Final Average F1 Scores ====")
    for k in K_LIST:
        avg = total_f1_scores[k] / n
        print(f"F1@{k}: {avg:.4f}")

    # === 保存结果 ===
    with open("~/deepseek_predictions_multiK.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"\n 已保存 {len(results)} 条预测结果到 deepseek_predictions_multiK.jsonl")

"""
def load_dataset(path: str, max_samples: int = None) -> List[Dict]:
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            dataset.append(json.loads(line))
    return dataset

def main_evaluation_flow(
    data_path: str,
    k_list: List[int],
    max_samples: int = None,
    template_name="zero_shot_technical",
    max_k=5
    ) -> Dict[str, float]:



    dataset = load_dataset(data_path, max_samples)
    #print("The first data example:",dataset[0])
    if not dataset:
        raise ValueError("Fail to load dataset")


    results = {
        template_name: {k: [] for k in k_list}
        for template_name in PROMPT_TEMPLATES
    }

    for sample in dataset:
        text = sample["input"]
        true_keywords = sample["output"]


        for template_name, template in PROMPT_TEMPLATES.items():
            try:

                predicted_keywords = call_keyword_extraction_api(
                    text=text,
                    template_name="zero_shot_technical",
                    max_k=np.max(k_list) # Extract by max_{k} and truncate later
                )

                # For every k value
                for k in k_list:
                    f1 = evaluate_f1_at_k(
                        true_keywords=true_keywords,
                        predicted_keywords=predicted_keywords[:k], # truncate the first k keywords
                        k=k,
                        prompt_info=template_name
                    )
                    results[template_name][k].append(f1)

            except RetryError as e:
                logger.error(f"sample skip（template={template_name}）: {str(e)}")
                continue

    final_stats = {}
    for template_name, k_results in results.items():
        final_stats[template_name] = {
            k: sum(scores) / len(scores) if scores else 0
            for k, scores in k_results.items()
        }

    return final_stats

#if __name__ == "__main__":
    #evaluate()
#f1 = evaluate_f1_at_k(predicted_keywords=['Thermalization in nuclear reactions','Nuclear Theory'], true_keywords=['thermalization','Nuclear Theory'],k=2)
#print(f1)

if __name__ == "__main__":
    settings = {
        "data_path": "~/train_instruction.json",  # Absolute Directory
        "k_list": [5, 15, 25, 35],
        "max_samples": 100 # limit the number of examples
    }

    # Another template (In Mandarin)
    custom_templates = {
        "technical": """
        你是一个领域专家，请从以下文本中提取最多{max_k}个技术术语：
        文本：\"\"\"{text}\"\"\"
        要求：
        - 必须是名词短语
        - 按重要性降序排列
        """,
        "simple": "提取以下文本的{max_k}个关键词：\"\"\"{text}\"\"\""
    }

    stats = main_evaluation_flow(
        **settings,
        template_name="zero_shot_technical"
    )

    print("Evaluation results（Mean F1@K）：")
    for template, scores in stats.items():
        print(f"\ntemplate【{template}】")
        for k, f1 in scores.items():
            print(f" F1@{k}: {f1:.3f}")
