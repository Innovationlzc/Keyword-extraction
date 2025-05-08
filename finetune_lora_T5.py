from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import time
# 加载模型和分词器
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 添加 LoRA 配置
lora_config = LoraConfig(
    r=8,           # LoRA 秩
    lora_alpha=32,  # 缩放系数
    target_modules=["q", "v"],  # 针对 T5 的注意力模块
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 检查可训练参数量

# 加载数据集
dataset = load_dataset("json", data_files="train_instruction.json")
def preprocess(examples):
    inputs = ["extract keywords: " + text for text in examples["input"]]
    # targets = [", ".join(keywords) for keywords in examples["keywords"]]
    targets = examples['output']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True,padding="max_length")
    labels = tokenizer(targets, max_length=32, truncation=True,padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_dataset = dataset.map(preprocess, batched=True)


# 训练配置
training_args = TrainingArguments(
    output_dir="./T5_base",
    per_device_train_batch_size=16,
    num_train_epochs=80,
    learning_rate=1e-2,
    fp16=True,
    logging_steps=100,
    save_steps=500,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    # train_dataset=tokenized_dataset,
)

start_time = time.time()           # 记录开始时间
trainer.train()
end_time = time.time()             # 记录结束时间

elapsed = end_time - start_time
print(f"Training finished in {elapsed/60:.2f} minutes "
      f"({elapsed:.1f} seconds).")



model.save_pretrained("./lora_T5_model_lr1e-2_epoch80_test")