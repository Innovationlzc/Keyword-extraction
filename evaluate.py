import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util


# === 路径设置 ===
input_path = "deepseek_predictions_cleaned_final.jsonl"  # 替换为你清洗后文件路径
K_LIST = [5, 15, 25, 35]  # 支持多个K值
encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")


# === 核心评估函数 ===
def compute_metrics(pred, gt, k):
    pred_set = set(pred[:k])
    gt_set = set(gt)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1


def compute_avg_similarity(pred_list, gt_list):
    if not pred_list or not gt_list:
        return 0.0
    embeddings = encoder.encode(pred_list + gt_list, convert_to_tensor=True)
    pred_embeds = embeddings[:len(pred_list)]
    gt_embeds = embeddings[len(pred_list):]
    sims = util.pytorch_cos_sim(pred_embeds, gt_embeds)
    max_sims = sims.max(dim=1).values
    return max_sims.mean().item()

# === 主评估逻辑 ===
def evaluate(path, k_list):
    total_metrics = {k: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for k in k_list}
    similarity_total = 0.0
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gt = [kw.strip().lower() for kw in item.get("ground_truth", [])]
            pred = [kw.strip().lower() for kw in item.get("prediction_list", [])]

            if not gt or not pred:
                continue

            count += 1
            similarity_total += compute_avg_similarity(pred, gt)
            for k in k_list:
                p, r, f1 = compute_metrics(pred, gt, k)
                total_metrics[k]["precision"] += p
                total_metrics[k]["recall"] += r
                total_metrics[k]["f1"] += f1

    # 输出平均值
    avg_similarity = similarity_total / count if count > 0 else 0.0
    print(f"\n 评估样本数：{count}")
    print(f"\n 所有预测关键词的平均语义相似度（与最接近的GT对比）: {avg_similarity:.4f}")
    for k in k_list:
        avg_p = total_metrics[k]["precision"] / count
        avg_r = total_metrics[k]["recall"] / count
        avg_f1 = total_metrics[k]["f1"] / count
        print(f"\n K = {k}")
        print(f"Precision@{k}: {avg_p:.4f}")
        print(f"Recall@{k}:    {avg_r:.4f}")
        print(f"F1@{k}:        {avg_f1:.4f}")




# === 执行入口 ===
if __name__ == "__main__":
    evaluate(input_path, K_LIST)