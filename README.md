# QMSum

## Data Analysis

- Average input token length: 16365
- Maximum input token length: 30164

- Average summary token length: 64
- Maximum summary token length: 171


![input_token_length_distribution](https://github.com/Sleepyhead01/QMSum/assets/69421538/8ae916b6-27c1-4787-9f8d-a786d7c9e364)

Token distribution of input


![summary_token_length_distribution](https://github.com/Sleepyhead01/QMSum/assets/69421538/9f41d24c-dd12-4088-9ea5-4a93c7ab2658)

Token distribution of summary

*Note: These values are with respect to `EleutherAI/gpt-neox-20b` tokenizer

Code for this analysis present [here](https://github.com/Sleepyhead01/QMSum/blob/main/mpt-storywriter/token_len_analysis.py)

## Experiments
Python 3.8 is recommended.
Make a python virtual env and install the dependencies from `requirements.txt`

### MPT-Storywriter
- Can process up to 80k tokens (Input + Generation)

#### Generating with MPT-Storywriter

Set the  device map [here](https://github.com/Sleepyhead01/QMSum/blob/570abc33308f729ec42c1f6bb71d30386344185d/mpt-storywriter/mpt_generate.py#L58) according to the number of GPUs available in a node and parts of the model that you want to offload to cpu.

Write a job submission file by referring to [this](https://github.com/Sleepyhead01/QMSum/blob/main/mpt-storywriter/inf_gen.sh).

### long-former
- Can process up to ~100k tokens (Input + Generation)

#### Generating with long-former

The default script runs long former with instructions. To run without instructions uncomment [this](https://github.com/Sleepyhead01/QMSum/blob/9cd4684abdb869a743aa2b68b11ad46dbaad771f/long-former/generate_summary.py#L64) line and comment the next line. 

Write a job submission file by referring to [this](https://github.com/Sleepyhead01/QMSum/blob/main/mpt-storywriter/inf_gen.sh).

### LLaMA Adapter
- Can process up to 2k tokens (Input + Generation)

#### `Generating with LLaMA Adapter`

Clone the LLaMA Adapter [repo](https://github.com/OpenGVLab/LLaMA-Adapter)

Download LLaMA weights and LLaMa adapter weights.

Add the `generate_summary.py` script to the LLaMA-Adapter folder and replace the model weights path.

### LongLLaMA

Run `generate.py`

## Results

<img width="451" alt="Screenshot 2023-10-11 174219" src="https://github.com/Sleepyhead01/QMSum/assets/69421538/733f2ba0-4850-45fb-855b-490a68921bb9">


