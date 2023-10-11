import torch
from transformers import pipeline
import json
from transformers import LlamaTokenizer, AutoModelForCausalLM


# def generate_summary_with_instruction(summarizer, prompt):
#     tokenizer = LEDTokenizer.from_pretrained("hyesunyun/update-summarization-bart-large-longformer")    
#     tokenized_instruction = tokenizer.encode("Instruction:" + prompt['instruction'] + "\nSummary: ")
#     tokenized_convo = tokenizer.encode(prompt['input'])

#     convo_length= 15000-len(tokenized_instruction)
#     tokenized_prompt = tokenized_convo[:convo_length] + tokenized_instruction
#     prompt_ = tokenizer.decode(tokenized_prompt) 

#     result = summarizer(
#     prompt_,
#     # min_length=216,
#     # max_length=512,
#     min_length=16,
#     max_length=256,
#     no_repeat_ngram_size=3,
#     encoder_no_repeat_ngram_size=3,
#     repetition_penalty=3.5,
#     num_beams=4,
#     early_stopping=True,
# )    
#     result = result[0]
#     print(result)
#     return result['summary_text']

def generate_summary(model, prompt):
    # prompt = "My name is Julien and I like to"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    num_beams=1,
    last_context_length=1024,
    do_sample=True,
    temperature=1.0,
)
    # print()   
    result = tokenizer.decode(generation_output[0])
    print(result)
    return result

if __name__ == "__main__":

    tokenizer = LlamaTokenizer.from_pretrained("syzymon/long_llama_3b_v1_1")
    model = AutoModelForCausalLM.from_pretrained("syzymon/long_llama_3b_v1_1", 
                                                torch_dtype=torch.float32, 
                                                trust_remote_code=True)


    test = json.load(open('/home/millon/projects/def-fard/millon/LLaMA-Adapter/QMSum/processed_academic_data/alpaca_format_train.json'))
    test = test[:5]
    prompts = [x for x in test]

    results = []
    for prompt in prompts:
        results.append(generate_summary(model, prompt['input']))
        # results.append(generate_summary_with_instruction(summarizer, prompt))


    import pandas as pd
    test = pd.DataFrame(test)
    test['summary'] = results
    
    test.to_csv('inference_results_longllama.csv', index=False)

