# from llama import Tokenizer
import torch
import pandas as pd
import json
import  tqdm
from transformers import AutoTokenizer
# tokenizer = Tokenizer("/home/millon/projects/def-fard/millon/LLaMA-Adapter/llama_adapter_v2_multimodal7b/models/tokenizer.model")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


#load jsonl file as df
# path = '/home/millon/projects/def-fard/millon/QMSum/processed_academic_data/alpaca_format_test.json'
# df = pd.read_json('/home/millon/projects/def-fard/millon/LLaMA-Adapter/QMSum/processed_academic_data/train.jsonl', lines=True)
data = json.load(open('/home/millon/projects/def-fard/millon/QMSum/processed_academic_data/alpaca_format_train.json'))
texts = [x for x in data]
input_token_len = []
summary_token_len = []
maximum_input = 0
maximum_summary = 0
max_untokenized = 0
from tqdm import tqdm

for text in tqdm(texts):
    example = torch.tensor(tokenizer.encode(text["output"]), dtype=torch.int64)
    summary_token_len.append(len(example))
    maximum_summary = max(maximum_summary, len(example))

    prompt = PROMPT_DICT["prompt_input"].format(instruction=text['instruction'], input=text["input"])
    input_token_len.append(len(torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)))
    maximum_input = max(maximum_input, len(torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)))

    max_untokenized = max(max_untokenized, len(prompt))

print(f'Average input token length: {sum(input_token_len)/len(input_token_len)}')
print(f'Maximum input token length: {maximum_input}')

print(f'Average summary token length: {sum(summary_token_len)/len(summary_token_len)}')
print(f'Maximum summary token length: {maximum_summary}')

print(f'Maximum untokenized input length: {max_untokenized}')

import matplotlib.pyplot as plt
# plot the distribution of token lengths
plt.hist(input_token_len, bins=100)
plt.title('Input token length distribution')
plt.xlabel('Token length')
plt.ylabel('Number of examples')
plt.savefig('input_token_length_distribution.png')
plt.close()
#make a different plot for summary token length
plt.hist(summary_token_len, bins=100)
plt.title('Summary token length distribution')
plt.xlabel('Token length')
plt.ylabel('Number of examples')
plt.savefig('summary_token_length_distribution.png')

