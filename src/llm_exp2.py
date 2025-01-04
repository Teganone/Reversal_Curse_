import pandas as pd
from attr import define
import torch
from typing import List, Any, Dict, Tuple
from tqdm import tqdm

from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import random
# 加载环境变量
load_dotenv()

# 初始化 Azure OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment_name="gpt-4",
    model_name="gpt-4",
    temperature=0,
    max_tokens=50,
    max_retries=2,
)


df = pd.read_csv("./data/celebrity_relations/parent_child_pairs.csv")

UNKNOWN_STR = "Unknown"
# SYSTEM_PROMPT = f'''You are a helpful and terse assistant. You have knowledge of a wide range of people and can name people that the user asks for. If the answer is unknown or not applicable, answer with "{UNKNOWN_STR}"'''
SYSTEM_PROMPT = f'''You are a helpful assistant with knowledge of aboundant knowledge of a wide range of celebrities including their family relationships. When asked to find parent or child among the celebrations, you will provide the correct name of a person based on factual relationships between the celebrities. If the answer is unknown, answer with "{UNKNOWN_STR}".'''

SYSTEM_PROMPT = f'You are an expert when it comes to any type of celebrity, including but not limited to actors, singers, producers and others, and including their family relations. You answer questions concisely, with only the answer or {UNKNOWN_STR}'

@define
class ParentChildPair:
    child: str
    parent: str
    parent_type: str  # either 'mother' or 'father'

    def ask_for_parent(self) -> str:
        return f"Who is {self.child}'s {self.parent_type}?"

    def ask_for_child(self) -> str:
        # return f"Name a famous child of {self.parent}."
        return f'Which celebrity has a {self.parent_type} named {self.parent}?'

    def create_parent_query_chat_pair(self) -> list:
        return [
            self.ask_for_parent(),self.parent
        ]

    def create_child_query_chat_pair(self) -> list:
        return [
            self.ask_for_child(),self.child,
        ]


def flatten(list_of_lists: List[List]):
    return [item for sublist in list_of_lists for item in sublist]
few_shot_examples = flatten(
    [
        ParentChildPair("Malia Obama", "Barack Obama", "father").create_child_query_chat_pair(),
        ParentChildPair("Elon Musk", "Maye Musk", "mother").create_child_query_chat_pair(),
        ParentChildPair("Kathy Pratt", UNKNOWN_STR, "mother").create_parent_query_chat_pair(),
    ]
)

augment_can_reverse = []
correct_predictions=0
predicted_child_list = []
for index, row in tqdm(df.iterrows(),desc="process"):
    child, parent, parent_type, _, can_reverse = row
    pair = ParentChildPair(child, parent, parent_type)
    
    # query = f'Which celebrity has a {parent_type} named {parent}?'
    # query = f"Name a famous child of {parent}"
    query = pair.ask_for_child()
    prompt = SYSTEM_PROMPT+str(few_shot_examples)+query
    response = llm.invoke(prompt)
    child_prediction = response.content if hasattr(response, 'content') else response
    we_can_reverse = False
    if child_prediction in child or child in child_prediction:
        we_can_reverse = True
        correct_predictions+=1
        child_prediction = child
    elif child_prediction == UNKNOWN_STR:
        child_prediction = ''
    predicted_child_list.append(child_prediction)
    augment_can_reverse.append(we_can_reverse)
    print(f"Reverse {child},{parent},{parent_type},{child_prediction}\t{we_can_reverse}")
accuracy = correct_predictions / len(df)
print(f"Accuracy: {accuracy * 100:.2f}%")
df['augument_can_reverse'] = augment_can_reverse
df['aug_predicted_child'] = predicted_child_list
# df.to_csv("./data/celebrity_relations/parent_child_pairs_llm_prompt.csv",index=False)
df.to_csv("./data/celebrity_relations/parent_child_pairs_llm_which_prompt.csv",index=False)

