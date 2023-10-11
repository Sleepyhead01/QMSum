import torch
import transformers
from transformers import pipeline
import json
from transformers import AutoTokenizer, LEDTokenizer


def generate_summary_with_instruction(summarizer, prompt):
    tokenizer = LEDTokenizer.from_pretrained("hyesunyun/update-summarization-bart-large-longformer")    
    tokenized_instruction = tokenizer.encode("Instruction:" + prompt['instruction'] + "\nSummary: ")
    tokenized_convo = tokenizer.encode(prompt['input'])

    convo_length= 15000-len(tokenized_instruction)
    tokenized_prompt = tokenized_convo[:convo_length] + tokenized_instruction
    prompt_ = tokenizer.decode(tokenized_prompt) 

    result = summarizer(
    prompt_,
    min_length=16,
    max_length=256,
    no_repeat_ngram_size=3,
    encoder_no_repeat_ngram_size=3,
    repetition_penalty=3.5,
    num_beams=4,
    early_stopping=True,
)    
    result = result[0]
    print(result)
    return result['summary_text']

def generate_summary(summarizer, prompt):
    # with torch.autocast('cuda', dtype=torch.bfloat16):
    result = summarizer(prompt+"\nA short summary of the following conversation is:",
            # min_length=30, 
            # max_length=100)
            max_new_tokens=100,
            do_sample=True,
            use_cache=True)

    print(result)
    result = result[0]['generated_text']
    result = result[result.find("A short summary of the following conversation is:")+len("A short summary of the following conversation is:"):]
    return result

if __name__ == "__main__":
    print(50*'=')
    print("Multi Cpu")
    print(50*'=')

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    name = 'mosaicml/mpt-7b-storywriter'

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    config.max_seq_len = 30000 # (input + output) tokens can now be up to 83968
    print(f"Max Seq Len: {config.max_seq_len}")
    # config.init_device='cpu'

    hf_device_map = {'transformer.wte': 0,
                    'transformer.emb_drop': 0,
                    'transformer.blocks.0': 0,
                    'transformer.blocks.1': 0,
                    'transformer.blocks.2': 0,
                    'transformer.blocks.3': 1,
                    'transformer.blocks.4': 1,
                    'transformer.blocks.5': 1,
                    'transformer.blocks.6': 1,
                    'transformer.blocks.7': 1,
                    'transformer.blocks.8': 2,
                    'transformer.blocks.9': 2,
                    'transformer.blocks.10': 2,
                    'transformer.blocks.11': 2,
                    'transformer.blocks.12': 2,
                    'transformer.blocks.13': 3,
                    'transformer.blocks.14': 3,
                    'transformer.blocks.15': 3,
                    'transformer.blocks.16': 3,
                    'transformer.blocks.17': 3,
                    'transformer.blocks.18': 'cpu',
                    'transformer.blocks.19': 'cpu',
                    'transformer.blocks.20': 'cpu',
                    'transformer.blocks.21': 'cpu',
                    'transformer.blocks.22': 'cpu',
                    'transformer.blocks.23': 'cpu',
                    'transformer.blocks.24': 'cpu',
                    'transformer.blocks.25': 'cpu',
                    'transformer.blocks.26': 'cpu',
                    'transformer.blocks.27': 'cpu',
                    'transformer.blocks.28': 'cpu',
                    'transformer.blocks.29': 'cpu',
                    'transformer.blocks.30': 'cpu',
                    'transformer.blocks.31': 'cpu',
                    'transformer.norm_f': 'cpu'}    
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map=hf_device_map,
    )
    print(model.hf_device_map)

    test = json.load(open('/home/millon/projects/def-fard/millon/QMSum/processed_academic_data/alpaca_format_test.json'))
    # test = test[:5] 
    prompts = [x for x in test]

    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
    # pipe = pipeline('summarization', model=model, tokenizer=tokenizer)

    results = []
    for prompt in prompts:
        print('Input token length: ',len(torch.tensor(tokenizer.encode(prompt['input']), dtype=torch.int64)))
        results.append(generate_summary(pipe, prompt['input']))
        # results.append(generate_summary_with_instruction(summarizer, prompt))


    import pandas as pd
    test = pd.DataFrame(test)
    test['summary'] = results
    
    test.to_csv('results/inference_results_mpt_mcpu.csv', index=False)

