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

### MPT-Storywriter
- Can process up to 80k tokens (Input + Generation)

`Running MPT-Storywriter`

Set the  device map [here](https://github.com/Sleepyhead01/QMSum/blob/570abc33308f729ec42c1f6bb71d30386344185d/mpt-storywriter/mpt_generate.py#L58) according to the number of GPUs available in a node and parts of the model that you want to offload to cpu.

Write a job submission file by referring to [this](https://github.com/Sleepyhead01/QMSum/blob/main/mpt-storywriter/inf_gen.sh).


## Results

This is the results section.

