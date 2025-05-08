import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

# 加载模型和tokenizer
base_model = "ml6team/keyphrase-extraction-distilbert-inspec"
lora_model_path = "./lora_DB_model_lr1e-5_epoch50"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()

def predict_keyphrases(text):
    """对单个文本预测关键词"""
    tokens = text.split()
    
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    outputs = model(**inputs)
    predictions = np.argmax(outputs.logits.detach().numpy(), axis=2)[0]

    word_ids = inputs.word_ids()
    previous_word_idx = None
    keyphrase_words = []
    keyphrases = []

    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            if predictions[i] == 1:  # 假设1是关键词标签
                keyphrase_words.append(tokens[word_idx])
            elif keyphrase_words:
                keyphrases.append(" ".join(keyphrase_words))
                keyphrase_words = []
        previous_word_idx = word_idx

    if keyphrase_words:
        keyphrases.append(" ".join(keyphrase_words))

    return keyphrases

def process_test_file(input_file, output_file):
    """处理测试文件并保存预测结果"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            data = json.loads(line)
            input_text = data["input"]
            ground_truth = data["output"].split(',')
            
            # 预测关键词
            predictions = predict_keyphrases(input_text)
            
            # 构建输出数据
            result = {
                "input": input_text,
                "ground_truth": ground_truth,
                "prediction_list": predictions
            }
            
            # 写入新文件
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

# 使用示例
# input_test_file = "train_instruction.json"
# input_test_file = "test_cleaned.json"
input_test_file = "test_cleaned.json"  # 输入测试文件
output_result_file = "output/test_prediction_db_1e-5_epoch50.json"  # 输出结果文件

process_test_file(input_test_file, output_result_file)
print(f"预测完成，结果已保存到 {output_result_file}")