from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
# from datasets import load_dataset
import evaluate
import numpy as np
import json
from datasets import Dataset #load_metric

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    "ml6team/keyphrase-extraction-distilbert-inspec"
    )

def process_data(example):
    # 将关键词列表转换为token级别的标签
    text = example["input"]
    keywords = [kw.strip() for kw in example["output"].split(",")]
    
    # 分词并标记关键词
    tokens = text.split()
    labels = [0] * len(tokens)  # 0表示非关键词
    
    # 简单匹配关键词（实际应用中可能需要更复杂的匹配逻辑）
    for kw in keywords:
        kw_tokens = kw.split()
        for i in range(len(tokens) - len(kw_tokens) + 1):
            if tokens[i:i+len(kw_tokens)] == kw_tokens:
                labels[i:i+len(kw_tokens)] = [1] * len(kw_tokens)
    
    return {
        "tokens": tokens,
        "ner_tags": labels,
        "text": text,
        "keywords": keywords
    }

# 加载JSON数据
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:  # 明确指定utf-8编码
        data = [json.loads(line) for line in f]
    
    processed_data = [process_data(example) for example in data]
    return Dataset.from_dict({
        "tokens": [x["tokens"] for x in processed_data],
        "ner_tags": [x["ner_tags"] for x in processed_data],
        "text": [x["text"] for x in processed_data],
        "keywords": [x["keywords"] for x in processed_data]
    })

# 加载训练和验证数据
train_dataset = load_dataset("train_instruction.json")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 特殊token设置为-100
            if word_idx is None:
                label_ids.append(-100)
            # 每个单词的第一个token设置标签
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
# dataset = tokenized_train["train"].train_test_split(test_size=0.1)  # 10%作为验证集
# tokenized_train = dataset["train"]
# tokenized_eval = dataset["test"]

peft_config = LoraConfig(
    task_type="TOKEN_CLS",
    inference_mode=False,
    r=8,  # LoRA秩
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_lin", "k_lin", "v_lin"],  # DistilBERT的注意力模块
    modules_to_save=["pre_classifier", "classifier"]  # 需要完整微调的层
)

# seqeval = load_metric("seqeval")

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     # 定义标签映射（必须与数据标注一致！）
#     label_list = ["O", "B-KEY", "I-KEY"]  # 根据实际任务修改
#     # label_list = ["O", "B-KEY", "I-KEY"]  # 根据实际任务修改
    
#     true_predictions = [
#         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]

#     results = seqeval.compute(
#         predictions=true_predictions, 
#         references=true_labels
#     )
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }

# 加载模型
model = AutoModelForTokenClassification.from_pretrained(
    "ml6team/keyphrase-extraction-distilbert-inspec",
    num_labels=3,  # 确保与BIO标签数一致
    ignore_mismatched_sizes=True
)

# 应用LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./keyphrase_lora",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    eval_strategy="no",
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=50,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    # eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./lora_DB_model_lr1e-5_epoch50")


# 加载模型
# from peft import PeftModel

# loaded_model = AutoModelForTokenClassification.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec")
# loaded_model = PeftModel.from_pretrained(loaded_model, "./lora_keyphrase_model")