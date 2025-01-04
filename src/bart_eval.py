from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import os
import json
import torch
from tqdm import tqdm

# from reversal_curse.src.tasks.base_evaluator import evaluate_model_on_file
model = BartForConditionalGeneration.from_pretrained("./fine_tuned_bart_model")
tokenizer = BartTokenizer.from_pretrained("./fine_tuned_bart_model")


def get_prompts_targets(data: List[Dict], data_type: str) -> Tuple[List[str], List[str]]:
    prompts = [preprocess_prompt_for_eval(example["prompt"]) for example in data]
    targets = [preprocess_target_for_eval(example["completion"]) for example in data]
    return prompts, targets

def preprocess_prompt_for_eval(prompt: str) -> str:
    return prompt

def preprocess_target_for_eval(target: str) -> str:
    return target


def load_data(data_file: str) -> List[Dict]:
    if not os.path.exists(data_file):
        raise ValueError(f"Data file {data_file} does not exist")

    data = load_from_jsonl(data_file)
    # TODO: after refactor: sample randomly instead, otherwise might e.g. only evaluate on CoT realized examples
    # data = data[: max_samples]
    return data

def load_from_jsonl(file_name: str):
    with open(file_name, "r") as f:
        data = [json.loads(line) for line in f]
    return data

model_type = "bart"
def evaluate_model_on_file(data_file: str, data_type: str) -> Tuple[pd.DataFrame, Dict]:
    data = load_data(data_file)
    prompts, targets = get_prompts_targets(data, data_type)
    targets_lists = [[target] for target in targets]

    df = pd.DataFrame({"prompt": prompts, "target": targets})
    metrics = {}

    # for model, model_type in models:
    scores = cond_log_prob(tokenizer, model, prompts, targets_lists, absolute_normalization=True)
    completions = generate(prompts, tokenizer=tokenizer, model=model)
    accuracy, is_correct_list, sim_acc, similarity_list = evaluate_completions(completions, targets, threshold=0.6)
    # scores_single = [score[0] if len(score) == 1 else score for score in scores]
    scores_single = scores
    df[f"logprobs_{model_type}"] = scores_single
    df[f"completion_{model_type}"] = completions
    df[f"matched_{model_type}"] = is_correct_list
    df[f"similarity_{model_type}"] = similarity_list
    metrics[f"acc_{data_type}_{model_type}"] = accuracy
    metrics[f"acc_similarity_{data_type}_{model_type}"] = sim_acc


    sort_function = lambda x: (
        not x.startswith("prompt"),
        not x.startswith("target"),
        x.startswith("completion_"),
        x.startswith("logprobs_"),
        x.startswith("similarity_"),
        x.startswith("matched_"),
    )

    # added axis=1, otherwise it just is a table with columns and rows with the same labels and all nan afaict
    df = df.reindex(sorted(df.columns, key=sort_function), axis=1)
    return df, metrics



def cond_log_prob(tokenizer, model, prompts, targets_lists, absolute_normalization=True):
    """计算给定提示和目标的条件对数概率"""
    # 将输入和目标文本编码为模型输入
    inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    targets = tokenizer([target[0] for target in targets_lists], return_tensors="pt", padding="max_length", truncation=True, max_length=256)

    # 将输入传递给模型进行推理
    with torch.no_grad():
        outputs = model(**inputs, labels=targets["input_ids"])

    # 获取对数概率（loss），并计算条件对数概率
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # 获取目标的token ID
    target_ids = targets["input_ids"]

    # 获取每个目标 token 的对数概率
    scores = []
    for i, target_id_list in enumerate(target_ids):
        score = 0
        for j, target_id in enumerate(target_id_list):
            if target_id != tokenizer.pad_token_id:
                score += log_probs[i, j, target_id].item()
        scores.append(score)
    return scores


def generate(prompts, tokenizer, model):
    inputs = tokenizer(prompts, return_tensors="pt", max_length=256, truncation=True, padding="longest")
    # 使用生成方法生成输出
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)

    # 解码生成的 token IDs 为文本
    completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return completions

from sentence_transformers import SentenceTransformer, util
eval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def evaluate_completions(completions: List[str], targets: List[str], threshold:float):
    """Compute accuracy of completions using cosine similarity.
    """
    n_correct = 0
    is_correct_list = []
    cos_similarity_list = []
    for completion, target in zip(completions, targets):
        sim = evaluate_completion(completion, target)
        cos_similarity_list.append(sim)
        correct = False
        if sim>=threshold:
            correct = True
            n_correct += 1
        is_correct_list.append(correct)
    sim_accuracy = sum(cos_similarity_list)/len(cos_similarity_list)
    accuracy = n_correct / len(completions)
    return accuracy, is_correct_list, sim_accuracy, cos_similarity_list

def evaluate_completion(
    completion: str,
    target: str,
)->float:
    """Evaluate completion using cosine similarity vs the target.
    """
    target = target.strip()
    completion = completion.strip()
    sentences = [completion] + [target]
    embeddings = eval_model.encode(sentences)
    sim = util.cos_sim(embeddings[0], embeddings[1])
    return float("{0:.4f}".format(sim.tolist()[0][0]))


KEYS_WE_CARE_ABOUT = [
    "p2d_reverse_prompts_test",
    "both_prompts_test",
    "p2d_prompts_test",
    "d2p_prompts_test",
    "d2p_reverse_prompts_test",
    "p2d_reverse_prompts_test_randomized",
    "d2p_reverse_prompts_test_randomized",
]

metrix_list = []
for data_file_name in tqdm(KEYS_WE_CARE_ABOUT):
    data_file = f"./data/reverse_experiments/june_version_7921032488/{data_file_name}.jsonl"
    df, metrix = evaluate_model_on_file(data_file=data_file, data_type=data_file_name)
    df.to_csv(f'./data/reverse_experiments/june_version_7921032488/results/{model_type}_sim_{data_file_name}.csv',index=False)
    metrix_list.append(metrix)


# Combine all the metrics into a single DataFrame
all_metrics = []
for metrix in metrix_list:
    for key, value in metrix.items():
        all_metrics.append({"key": key, "acc_bart": value})

# Create a DataFrame for all metrics
metrics_df = pd.DataFrame(all_metrics)

# Save all the metrics into a CSV file
metrics_df.to_csv(f'./data/reverse_experiments/june_version_7921032488/results/{model_type}_sim_acc_summary.csv', index=False)




    
