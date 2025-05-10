#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Call DeepSeek twice with two different instructions and save the raw model
reply (无长度或关键词数量限制) under "prediction_list".

Prompt variants
  • zero_shot_technical
  • few_shot_standard
"""
import json
import time
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ========= 基本配置 ========= #
API_KEY   = "****"      # ← 换成自己的
BASE_URL  = "https://api.deepseek.com"
DATA_FILE = "train_instruction.json" # 输入 350 条数据
SAVE_DIR  = Path("output")       # 保存目录
SLEEP     = 0.5                      # 简单的速率限制

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# ========= 两种 prompt ========= #
ZERO_SHOT_TECHNICAL = """\
Extract all technical keywords or key-phrases that appear *verbatim* in the \
following academic abstract. Return them one per line, no extra explanation.
"""

FEW_SHOT_STANDARD = """\
Here are examples:
Text: "Deep learning is transforming computer vision."
Keywords:
deep learning
computer vision

Now extract all keywords or key-phrases from the text below (one per line):
"""

PROMPTS: Dict[str, str] = {
    # "zero_shot_technical": ZERO_SHOT_TECHNICAL,
    "few_shot_standard":   FEW_SHOT_STANDARD,
}

# ========= API 调用保持原样 ========= #
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
def generate_keywords(paragraph: str, instruction: str) -> str:
    system_prompt = "You are an AI assistant that extracts keywords from text."
    user_prompt   = f"{instruction}\n\n{paragraph}"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=1024  # 不限定长度
    )
    return response.choices[0].message.content.strip()


def load_dataset(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main() -> None:
    data = load_dataset(DATA_FILE)
    for mode, prompt in PROMPTS.items():
        out_file = SAVE_DIR / f"predictions_{mode}.jsonl"
        with out_file.open("w", encoding="utf-8") as w:
            for item in tqdm(data, desc=mode):
                try:
                    raw_reply = generate_keywords(item["input"], prompt)
                    rec = {
                        "input":            item["input"],
                        "ground_truth":     item.get("output"),
                        "prediction_list":  raw_reply  # 原样保存
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    time.sleep(SLEEP)
                except Exception as e:
                    print(f"[WARN] skip one: {e}")
        print(f"{mode} finished → {out_file}")

if __name__ == "__main__":
    main()
