import json
import time
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


# === 设置 ===
API_KEY = "*****"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

K_LIST = [5, 10, 15]  # 要评估的K值列表
MAX_EXAMPLES = 350  # 限制评估数量


# === API 调用函数 ===
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
def generate_keywords(paragraph: str, instruction: str):
    system_prompt = "You are an AI assistant that extracts keywords from text."
    user_prompt = f"{instruction}\n\n{paragraph}"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=128,
    )
    return response.choices[0].message.content.strip()


# === 评估函数：F1@K ===
def compute_f1_at_k(pred_keywords: List[str], gt_keywords: List[str], k: int) -> float:
    pred_set = set(pred_keywords[:k])
    gt_set = set(gt_keywords)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


# === 主评估流程 ===
def evaluate():
    with open("train_instruction.json", "r", encoding="utf-8") as f:
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
    with open("deepseek_predictions_multiK.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"\n 已保存 {len(results)} 条预测结果到 deepseek_predictions_multiK.jsonl")


if __name__ == "__main__":
    evaluate()
