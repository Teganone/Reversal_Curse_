import json
from datasets import Dataset
import evaluate
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
import torch
torch.cuda.empty_cache()
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 加载数据
with open('./data/reverse_experiments/june_version_7921032488/all_prompts_train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f.readlines()]

with open('./data/reverse_experiments/june_version_7921032488/validation_prompts.jsonl', 'r') as f:
    val_data = [json.loads(line) for line in f.readlines()]

# 创建Dataset对象
train_dataset = Dataset.from_dict({
    'input_text': [entry['prompt'] for entry in train_data],
    'target_text': [entry['completion'] for entry in train_data]
})

val_dataset = Dataset.from_dict({
    'input_text': [entry['prompt'] for entry in val_data],
    'target_text': [entry['completion'] for entry in val_data]
})
# # 查看数据
# print(train_dataset[0])
# print(val_dataset[0])



# 加载预训练 BART 模型和 Tokenizer
model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenizer 对数据进行编码
def preprocess_function(examples):
    inputs = [ex for ex in examples['input_text']]
    targets = [ex for ex in examples['target_text']]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # 目标编码
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    
    model_inputs['labels'] = labels
    return model_inputs

# 对数据集进行处理
train_encoded_dataset = train_dataset.map(preprocess_function, batched=True,remove_columns=train_dataset.column_names,load_from_cache_file=False)
val_encoded_dataset = val_dataset.map(preprocess_function, batched=True,remove_columns=train_dataset.column_names,load_from_cache_file=False)


training_args = TrainingArguments(
    # output_dir="./checkpoints",               # 输出路径
    # evaluation_strategy="no",          # 每个 epoch 后评估模型
    # save_strategy="no",
    # learning_rate=2e-5,                   # 学习率
    # per_device_train_batch_size=2,        # 每个设备的训练批次大小
    # # per_device_eval_batch_size=1,         # 每个设备的评估批次大小
    # # gradient_accumulation_steps=4,            # 梯度累积步数
    # num_train_epochs=2,                   # 训练的 epoch 数量
    # weight_decay=0.01,                    # 权重衰减
    # logging_dir='./logs',                 # 日志存储路径
    # logging_steps=200,                    # 每200步记录一次日志
    # # save_steps=600,                       # 每500步保存一次模型
    # # save_total_limit=2,                   # 保留最多两个检查点
    # load_best_model_at_end=True,          # 在训练结束时加载最佳模型
    # metric_for_best_model="eval_loss",    # 使用评估损失来选择最佳模型
    # gradient_checkpointing=False,  # 使用梯度检查点
    # fp16=False,
    output_dir="./checkpoints",               # 输出路径
    evaluation_strategy="steps",              # 每隔一定步数评估
    save_strategy="steps",                    # 每隔一定步数保存模型
    save_steps=500,                           # 每500步保存一次模型
    eval_steps=500,                           # 每500步评估一次模型
    logging_dir='./logs',                     # 日志存储路径
    logging_steps=200,                        # 每100步记录一次日志
    num_train_epochs=3,                       # 训练的 epoch 数量
    weight_decay=0.01,                        # 权重衰减
    per_device_train_batch_size=2,            # 每个设备的训练批次大小
    per_device_eval_batch_size=1,             # 每个设备的评估批次大小
    load_best_model_at_end=True,              # 在训练结束时加载最佳模型
    metric_for_best_model="eval_loss",        # 使用评估损失来选择最佳模型
    gradient_accumulation_steps=4,            # 梯度累积步数
    fp16=True,
)

import gc

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 在关键点调用此函数，如训练开始前、每个 epoch 开始时等
print_gpu_memory()

class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:  # 每100步清理一次
            print_gpu_memory()
            gc.collect()
            torch.cuda.empty_cache()
            print_gpu_memory()


# class SaveModelCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         if state.global_step % 200 == 0:  # 每100步保存一次
#             output_dir = f"./model_step_{state.global_step}"
#             kwargs['trainer'].save_model(output_dir)
#             print(f"模型已保存到 {output_dir}")



metric = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 解码预测结果和标签
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 计算 BLEU 分数
    result = metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    return result



# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, max_length=512)

# 初始化 Trainer 并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded_dataset,
    eval_dataset=val_encoded_dataset,
    # callbacks=[SaveModelCallback()],
    callbacks=[MemoryCleanupCallback()],
    # data_collator = data_collator,
    compute_metrics=compute_metrics,  # 添加评估指标
)

# # 开始训练
# trainer.train()

# # 保存模型和 tokenizer
# model.save_pretrained("./fine_tuned_bart_model")
# tokenizer.save_pretrained("./fine_tuned_bart_model")
try:
    trainer.train()
    print("train finished!")
except Exception as e:
    print(f"训练过程中遇到错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("尝试保存最终模型...")
    trainer.save_model("./fine_tuned_bart_model_val")
    tokenizer.save_pretrained("./fine_tuned_bart_model_val")
    print("最终模型保存完成。")