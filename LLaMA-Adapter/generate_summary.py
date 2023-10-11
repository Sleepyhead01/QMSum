# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import LLaMA, ModelArgs, Tokenizer, Transformer

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
    "sub_summary_prompt_dummy": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nSummarize the following conversation.\n\n### Input:\n\n### Response:"
    ),
    "sub_summary_prompt": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nSummarize the following conversation.\n\n### Input:{input}\n\n### Response:"
    ),
}


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    quantizer: bool=False,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    print(model)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(adapter_checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def generate_summary(generator, prompt):
    tokenizer = Tokenizer(model_path="/home/millon/projects/def-fard/millon/LLaMA-Adapter/llama_adapter_v2_multimodal7b/models/tokenizer.model") #replace with path to tokenizer
    tokenized_content = tokenizer.encode(prompt['input'], bos=True, eos=True)
    tokenized_instruction_len = len(tokenizer.encode(PROMPT_DICT["sub_summary_prompt_dummy"].format_map(tokenized_content), bos=True, eos=True))

    chunks = []
    context_len = 400-tokenized_instruction_len
    for i in range(0, len(tokenized_content), context_len):
        convo = tokenizer.decode(tokenized_content[i:i + context_len])
        chunk = PROMPT_DICT['sub_summary_prompt'].format(input=convo)
        chunks.append(chunk)

    # Summarize each chunk
    summaries = ""
    for chunk in chunks:
        print(50*'*')
        print("length of chunk", len(tokenizer.encode(chunk, bos=True, eos=True)))
        print('*****************   CHUNK   *************************')
        print(chunk)
        #remove all words before "Input" in chunk
        summary = generator.generate([chunk], max_gen_len=700, temperature=0.1, top_p=0.75)
        print('*****************   SUMMARY RAW   *************************')
        print(summary)
        summary = summary[0]
        summary = summary[summary.find("Response:")+len("Response:"):]
        print('*****************   SUMMARY   *************************')
        print(summary)
        summaries+=summary
        summaries+="\n"

    print(50*'*')
    print("length of summaries", len(tokenizer.encode(summaries, bos=True, eos=True)))
    print(summaries)

    final_prompt = tokenizer.encode(summaries, bos=True, eos=True)
    final_prompt = tokenizer.decode(final_prompt[:500-tokenized_instruction_len])
    final_prompt = PROMPT_DICT["prompt_input"].format_map({"instruction": prompt['instruction'], "input": final_prompt})
    print('*****************   ERROR   *************************')

    print(final_prompt)
    generation = generator.generate([final_prompt], max_gen_len=800, temperature=0.1, top_p=0.75)
    generation = generation[0]

    print('*****************   FINAL SUMMARY   *************************')
    print(generation)
    generation = generation[generation.find("Response:")+len("Response:"):]
    return generation

    

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    temperature: float = 0.1,
    top_p: float = 0.75,
    # max_seq_len: int = 512,
    max_seq_len: int = 600,
    max_batch_size: int = 58,
    quantizer: bool = False,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size, quantizer)
    test = json.load(open('/home/millon/projects/def-fard/millon/LLaMA-Adapter/QMSum/processed_academic_data/alpaca_format_test.json'))
    # test = test[:5]
    prompts = [x for x in test]

    results = []
    for prompt in prompts:
        results.append(generate_summary(generator, prompt))

    import pandas as pd
    test = pd.DataFrame(test)
    test['summary'] = results
    
    test.to_csv('inference_results_500_1000_500_1000.csv', index=False)


if __name__ == "__main__":
    fire.Fire(main)
