import torch
from transformers import pipeline
import json
from transformers import LEDTokenizer


def generate_summary_with_instruction(summarizer, prompt):
    tokenizer = LEDTokenizer.from_pretrained("hyesunyun/update-summarization-bart-large-longformer")    
    tokenized_instruction = tokenizer.encode("Instruction:" + prompt['instruction'] + "\nSummary: ")
    tokenized_convo = tokenizer.encode(prompt['input'])

    convo_length= 15000-len(tokenized_instruction)
    tokenized_prompt = tokenized_convo[:convo_length] + tokenized_instruction
    prompt_ = tokenizer.decode(tokenized_prompt) 

    result = summarizer(
    prompt_,
    # min_length=216,
    # max_length=512,
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
    result = summarizer(
    prompt,
    min_length=216,
    max_length=512,
    no_repeat_ngram_size=3,
    encoder_no_repeat_ngram_size=3,
    repetition_penalty=3.5,
    num_beams=4,
    early_stopping=True,
)    
    result = result[0]
    print(result)
    return result['summary_text']

if __name__ == "__main__":
    hf_name ="hyesunyun/update-summarization-bart-large-longformer"

    summarizer = pipeline(
        "summarization",
        hf_name,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True, 
        max_length=16384,
    )

    test = json.load(open('/home/millon/projects/def-fard/millon/LLaMA-Adapter/QMSum/processed_academic_data/alpaca_format_test.json'))
    # test = test[:5]
    prompts = [x for x in test]

    results = []
    for prompt in prompts:
        # results.append(generate_summary(summarizer, prompt['input']))
        results.append(generate_summary_with_instruction(summarizer, prompt))


    import pandas as pd
    test = pd.DataFrame(test)
    test['summary'] = results
    
    test.to_csv('inference_results_long_former_i_.csv', index=False)

