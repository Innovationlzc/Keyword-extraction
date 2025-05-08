import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
import torch

# 加载基础模型和 LoRA 微调后的权重
base_model = "t5-small"  # 或 "t5-base"
lora_model_path = "./lora_T5_model_lr1e-2_epoch80_test"  # 替换为你的 LoRA 微调路径

# 加载 Tokenizer 和基础模型
tokenizer = T5Tokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model)
device = "cuda" if torch.cuda.is_available() else "cpu"

#
# tokenizer = T5Tokenizer.from_pretrained(base_model)          # 建议和模型同目录
# model = T5ForConditionalGeneration.from_pretrained(lora_model_path).to(device)
# model.eval()
# 加载 LoRA 权重（如果已微调）
model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()
model.to(device)


def extract_keyphrases(text):
    input_text = "extract keywords: " + text  # T5需要任务前缀
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=64,
        num_beams=5,
        early_stopping=True
    )
    keyphrases = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [kp.strip() for kp in keyphrases.split(",") if kp.strip()]

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
            predictions = extract_keyphrases(input_text)
            
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
input_test_file = "train_instruction.json"  # 输入测试文件
output_result_file = "output/train_prediction_T5_base_lr1e-4_epoch80.json"  # 输出结果文件

process_test_file(input_test_file, output_result_file)
print(f"预测完成，结果已保存到 {output_result_file}")