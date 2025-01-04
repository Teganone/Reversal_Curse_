import pandas as pd
from attr import define
from transformers import BertTokenizer, BertForQuestionAnswering,BertForMaskedLM
import torch
from typing import List, Any, Dict, Tuple
from tqdm import tqdm

from langchain_openai import AzureChatOpenAI
import os
import json
from dotenv import load_dotenv
import random
# 加载环境变量
load_dotenv()

# 初始化 Azure OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment_name="gpt-4o",
    model_name="gpt-4o",
    temperature=0,
    max_tokens=400,
    max_retries=2,
)

tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')

UNKNOWN_STR = "Unknown"
# SYSTEM_PROMPT = f'''You are a helpful assistant with knowledge of aboundant knowledge of a wide range of celebrities including their family relationships. When asked to find parent or child among the celebrations, you will provide the correct name of a person based on factual relationships between the celebrities. If the answer is unknown, answer with "{UNKNOWN_STR}".'''
SYSTEM_PROMPT = f'''You are a helpful assistant with knowledge of aboundant knowledge of a wide range of celebrities including their family relationships. When asked to find parent or child among the celebrations, you will provide the correct name of a person based on factual relationships between the celebrities. The answer should be concise as the correct name. If the answer is unknown, answer with "{UNKNOWN_STR}".'''

@define
class ParentChildPair:
    child: str
    parent: str
    parent_type: str  # either 'mother' or 'father'

    def ask_for_parent(self) -> str:
        return f"Who is {self.child}'s {self.parent_type}?"

    def ask_for_child(self) -> str:
        return f"Name a child of {self.parent}."

    def create_parent_query_chat_pair(self) -> list:
        return [
            self.ask_for_parent(),self.parent
        ]

    def create_child_query_chat_pair(self) -> list:
        return [
            self.ask_for_child(),self.child,
        ]

def ask_bert(question, context=None):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt", max_length=512, truncation="only_second")  # 优先截断超长context文本)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer_ids = input_ids[0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_ids,skip_special_tokens=True)
    return answer

    

def check_reverse_predictions(df):
    predicted_child_list = []
    bert_can_reverse_list = []
    correct_predictions=0
    for index, row in tqdm(df.iterrows()):
        child, parent, parent_type, _, can_reverse = row
        pair = ParentChildPair(child, parent, parent_type)
        
        question_reverse = f"Name a famous child of {parent}"
        # prompt = SYSTEM_PROMPT +f"Who is {child}?"
        prompt = f"You are a helpful assistant with knowledge of parent-child relationships. Please provide detailed information especially name of {child}'s parents, children, and family relations."
        response = llm.invoke(prompt)
        context = response.content if hasattr(response, 'content') else response
        # print(context)
        predicted_child = ask_bert(question_reverse, context)
        bert_can_reverse = False
        if child in predicted_child or predicted_child in child:
            correct_predictions += 1
            predicted_child = child
            bert_can_reverse = True
        # else:
        #     predicted_child = ''
        bert_can_reverse_list.append(bert_can_reverse)
        predicted_child_list.append(predicted_child)
        print(f"Reverse {child},{parent},{parent_type},{predicted_child}\t{bert_can_reverse}")

    df['bert_can_reverse'] = bert_can_reverse_list
    df['bert_predicted_child'] = predicted_child_list
    accuracy = correct_predictions / len(df)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return df


if __name__ == "__main__":
    df = pd.read_csv("./data/celebrity_relations/parent_child_pairs.csv")
    df = check_reverse_predictions(df)
    df.to_csv("./data/celebrity_relations/parent_child_pairs_bert.csv",index=False)

   