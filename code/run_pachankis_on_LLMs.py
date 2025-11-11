import torch
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper_functions import stigmas_list, concealability_classification, course_classification, disruptiveness_classification,aesthetics_classification, origin_classification, peril_classification, extract_value

#Code for prompting models with similar questions as Pachankis et al. 2018 and obtaining rankings across 93 stigmas
#Data used in Experiment 1, 2, and 3 in "Identifying Features Associated with Bias Against Stigmatized Groups in LLM Outputs and Guardrail Model Mitigation"


torch.cuda.empty_cache()
from huggingface_hub import login, logout
login("TOKEN HERE") 

#device = "auto"
granite_model = "ibm-granite/granite-3.0-8b-instruct"
tokenizer_g = AutoTokenizer.from_pretrained(granite_model)
model_g = AutoModelForCausalLM.from_pretrained(granite_model, device_map="auto")

mistral_model = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer_m = AutoTokenizer.from_pretrained(mistral_model)
model_m = AutoModelForCausalLM.from_pretrained(mistral_model, torch_dtype=torch.float16, device_map="auto")

llama_model = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer_l = AutoTokenizer.from_pretrained(llama_model)
model_l = AutoModelForCausalLM.from_pretrained(llama_model, torch_dtype=torch.float16, device_map="auto")

def run_granite(prompt):
    model_g.eval()

    chat = [
        { "role": "user", "content": prompt, "temperature" : 0.2},
    ]
    chat = tokenizer_g.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # tokenize the text
    input_tokens = tokenizer_g(chat, return_tensors="pt").to(model_g.device)
    # generate output tokens
    output = model_g.generate(**input_tokens, 
                        do_sample = True,
                        max_new_tokens=40)
    # decode output tokens into text
    output = tokenizer_g.batch_decode(output)

    cleaned_output = extract_value(str(output), 'granite')

    return cleaned_output

def run_mistral(prompt):
    #move these out of the run function, it keeps reloading them each time
    model_m.eval()

    chat = [
        { "role": "user", "content": prompt, "temperature" : 0.2},
    ]
    chat = tokenizer_m.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # tokenize the text
    input_tokens = tokenizer_m(chat, return_tensors="pt").to(model_m.device)
    # generate output tokens
    output = model_m.generate(**input_tokens, 
                              do_sample = True,
                        max_new_tokens=100)
    # decode output tokens into text
    output = tokenizer_m.batch_decode(output)
  
    #clean output to be just the number, unsure, or improper value
    cleaned_output = extract_value(str(output), 'mistral')

    return cleaned_output

def run_llama(prompt):

    chat = [
        { "role": "user", "content": prompt, "temperature" : 0.1},
    ]
    chat = tokenizer_l.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # tokenize the text
    input_tokens = tokenizer_l(chat, return_tensors="pt").to(model_l.device)
    # generate output tokens
    output = model_l.generate(**input_tokens, 
                              do_sample = True,
                        max_new_tokens=100)
    # decode output tokens into text
    output = tokenizer_l.batch_decode(output)
    cleaned_output = extract_value(output[0], 'llama')

    return cleaned_output

def generate_all_data(stigmas, iterations, model):

    if model == 'mistral':
        run_model = run_mistral
    
    if model == 'granite':
        run_model = run_granite

    if model == 'llama':
        run_model = run_llama

    
    visibilities = {k:[] for k in stigmas}
    course = {k:[] for k in stigmas}
    disrupt = {k:[] for k in stigmas}
    aesthetics = {k:[] for k in stigmas}
    origin = {k:[] for k in stigmas}
    peril = {k:[] for k in stigmas}


    for stigma in stigmas:
        for i in range(0,iterations):
            visibilities[stigma].append(run_model(concealability_classification(stigma)))
            course[stigma].append(run_model(course_classification(stigma)))
            disrupt[stigma].append(run_model(disruptiveness_classification(stigma)))
            aesthetics[stigma].append(run_model(aesthetics_classification(stigma)))
            origin[stigma].append(run_model(origin_classification(stigma)))
            peril[stigma].append(run_model(peril_classification(stigma)))
        print(stigma, 'done')
    
    return list(visibilities.values()), list(course.values()), list(disrupt.values()), list(aesthetics.values()), list(origin.values()), list(peril.values())

#obtain rankings for each dimension of the 93 stigmas with specific amount of iterations and model
#then save to CSV as necessary
visibilities, course, disrupt, aesthetics, origin, peril = generate_all_data(stigmas_list, 2 ,'granite')
