import pandas as pd
import torch
import numpy as np
from ast import literal_eval
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
#access to Granite transformers may require login
from huggingface_hub import login, logout
login("TOKEN HERE") 

#Used in Experiment 2 and 3 in "Identifying Features Associated with Bias Against Stigmatized Groups in LLM Outputs and Guardrail Model Mitigation"

#downstream implications based on IBM's SocialStigmaQA Benchmark
#Mistral, Llama, and Granite: How do they respond to these prompts?
df = pd.read_csv("hf://datasets/ibm/SocialStigmaQA/yes_no/full_dataset_yes_no.csv")
torch.cuda.empty_cache() #managing GPU cache

#import Granite
granite_model = "ibm-granite/granite-3.0-8b-instruct"
tokenizer_g = AutoTokenizer.from_pretrained(granite_model)
granite = AutoModelForCausalLM.from_pretrained(granite_model, device_map="auto")

def run_granite(prompt):
    message = [
        { "role": "user", "content": prompt, "temperature" : 0.1},
    ]
    chat = tokenizer_g.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    input_tokens = tokenizer_g(chat, return_tensors="pt").to(granite.device)
    output = granite.generate(**input_tokens, 
                        do_sample = True,
                        max_new_tokens=40)
    output = tokenizer_g.batch_decode(output)

    #clean output
    match = re.search(r'<\|end_of_role\|>\s*(yes|no)\b', str(output), re.IGNORECASE)
    if match:
        return match.group(1).lower()  # Return 'yes' or 'no' in lowercase
    else:
        return 'improper output'  # No match found

#import Llama
llama_model = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer_l = AutoTokenizer.from_pretrained(llama_model)
llama = AutoModelForCausalLM.from_pretrained(llama_model, torch_dtype=torch.float16, device_map="auto")

def run_llama(prompt):
    message = [
        { "role": "user", "content": prompt, "temperature" : 0.1},
    ]
    chat = tokenizer_l.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    input_tokens = tokenizer_l(chat, return_tensors="pt").to(llama.device)
    output = llama.generate(**input_tokens, 
                              do_sample = True,
                        max_new_tokens=100)
    output = tokenizer_l.batch_decode(output)
    split = re.search(r'(?<=<\|start_header_id\|>assistant<\|end_header_id\|>\n\n)(.*)', output[0], re.IGNORECASE)

    if split:
        new = split.group(0)
        match = re.search(r'\b(yes|no)\b', new, re.IGNORECASE)
        if match:
            return match.group(0).lower()
    else:
        print('improper output')
        return 'improper output'


#import Mistral
mistral_model = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer_m = AutoTokenizer.from_pretrained(mistral_model)
mistral = AutoModelForCausalLM.from_pretrained(mistral_model, torch_dtype=torch.float16, device_map="auto")

def run_mistral(prompt):
    message = [
        { "role": "user", "content": prompt, "temperature" : 0.1},
    ]
    chat = tokenizer_m.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    input_tokens = tokenizer_m(chat, return_tensors="pt").to(mistral.device)
    output = mistral.generate(**input_tokens, 
                              do_sample = True,
                        max_new_tokens=100)
    output = tokenizer_m.batch_decode(output)

    #clean output
    match = re.search(r'\[/INST\](.*?)(\b(yes|no)\b)', str(output), re.IGNORECASE)
    
    if match:
        cleaned =  match.group(3).lower()  # Return 'yes' or 'no' in lowercase
    else:
        cleaned = 'improper output'  # No match found
    
    return output, cleaned

mistral_answers = []
llama_answers = []
granite_answers = []
mistral_answers_cleaned = []
prompts = []

print("starting question asking")
for i in range(0, len(df)):
    prompt = df['prompt'][i]
    new_prompt = prompt.replace("Answer with yes/no/can't tell", "Answer first only with yes or no, then provide your reasoning")

    output, cleaned = run_mistral(prompt)
    if i == 1:
        print(output)
        print(cleaned)
    mistral_answers.append(output)
    mistral_answers_cleaned.append(cleaned)
    prompts.append(safe_prompt)

#collect if the given answer is biased (does it equal the biased answer column in socialstigmaqa)
biased_answers = []

for i in range(0,len(df)):
    #replace mistral answers with whichever model was ran
    if mistral_answers[i] == df['biased answer']:
        biased_answers.append(1)
    else:
        biased_answers.append(0)

data = {'stigma' : df['stigma'], "prompt": prompts, "prompt style": df['prompt style'], 'model answer': mistral_answers, 'ssqa-baised answer': df['biased answer'], 'biased answer': biased_answers}
df_mistral = pd.DataFrame(data)
#save to df as necessary, and repeat with the other models




