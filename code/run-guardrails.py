import warnings
import os
import pandas as pd

warnings.filterwarnings('ignore')
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import math
import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification
import nltk
import ssl
from mistralai import Mistral

#Used in Experiment 3 of "Identifying Features Associated with Bias Against Stigmatized Groups in LLM Outputs and Guardrail Model Mitigation"

#downstream implications based on IBM's SocialStigmaQA Benchmark
#import dataset
df = pd.read_csv("hf://datasets/ibm/SocialStigmaQA/yes_no/full_dataset_yes_no.csv")

#### IBM's Granite Guardian ####
model_path_name = "ibm-granite/granite-guardian-3.0-2b"
safe_token = "No"
unsafe_token = "Yes"
nlogprobs = 20
device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(model_path_name) #device_map="auto")
model = model.to(model.device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path_name)

def get_probablities(logprobs):
    safe_token_prob = 1e-50
    unsafe_token_prob = 1e-50
    for gen_token_i in logprobs:
        for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]):
            decoded_token = tokenizer.convert_ids_to_tokens(index)
            if decoded_token.strip().lower() == safe_token.lower():
                safe_token_prob += math.exp(logprob)
            if decoded_token.strip().lower() == unsafe_token.lower():
                unsafe_token_prob += math.exp(logprob)

    probabilities = torch.softmax(
        torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]), dim=0
    )

    return probabilities

def parse_output(output, input_len):
    label, prob_of_risk = None, None
    if nlogprobs > 0:

        list_index_logprobs_i = [torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                                 for token_i in list(output.scores)[:-1]]
        if list_index_logprobs_i is not None:
            prob = get_probablities(list_index_logprobs_i)
            prob_of_risk = prob[1]

    res = tokenizer.decode(output.sequences[:,input_len:][0],skip_special_tokens=True).strip()
    if unsafe_token.lower() == res.lower():
        label = unsafe_token
    elif safe_token.lower() == res.lower():
        label = safe_token
    else:
        label = "Failed"

    return label, prob_of_risk.item()

def detect_risk(user_text, harm_type):
    messages = [{"role": "user", "content": user_text}]

    #different types of harms, pre-detailed and trained by IBM
    #guardian_config = {"risk_name": "harm"} #default
   # guardian_config = {"risk_name": "social_bias"}
   # guardian_config = {"risk_name": "profanity"}
   # guardian_config = {"risk_name": "unethical_behavior"} #test this next
    #guardian_config = {"risk_name": "violence"} 
   # guardian_config = {"risk_name": "sexual_content"}

    guardian_config = {"risk_name": harm_type}

    input_ids = tokenizer.apply_chat_template(messages, guardian_config = guardian_config, add_generation_prompt=True, return_tensors="pt").to(model.device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True,
        )

    label, prob_of_risk = parse_output(output, input_len)
    #print("harm type:",guardian_config)
    #print(user_text)
    #print(f"\n# risk detected? : {label}")
    #print(f"# probability of risk: {prob_of_risk:.3f}") 
    return label, prob_of_risk

IBM_example = "How can I figure out the pin code to a phone?" #For social bias, this has very low risk. Makes sense. For default harm its much higher


def granite_guardian(harm_type):

    granite_guardian_label = []
    granite_guardian_probability= []
    prompts = []

    print("starting question asking")
    for i in range(10000,len(df)):
        prompt = df['prompt'][i]
    
        if i == 10100:
            print("200 left")
        if i == 10200:
            print("100 left")
        new_prompt = prompt.replace("Answer with yes/no/can't tell", "Answer first only with yes or no, then provide your reasoning")
        prompts.append(new_prompt)
        label, prob = detect_risk(new_prompt, harm_type)
   
        granite_guardian_label.append(label)
        granite_guardian_probability.append(round(prob, 5))

    return granite_guardian_label, granite_guardian_probability, prompts

#harm_label, harm_probability, prompts = granite_guardian("harm")
#harm_columns = {"prompt": prompts, "risk of harm (label)?" : harm_label, "probability of harm?" :harm_probability }

social_bias_label, social_bias_probability, prompts = granite_guardian("social_bias")
social_bias_columns = {"prompt": prompts, "risk of social bias (label)?" : social_bias_label, "probability of social bias?" : social_bias_probability }

df_bias = pd.DataFrame(social_bias_columns)
#save df as necessary
print('df done')


#### Meta's Llama Guard ####
from huggingface_hub import login, logout
#log into hugging_face

print("loading tokenizer and model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B", torch_dtype=torch.bfloat16, device_map="auto")

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def run_llama_guard():
    guard_outputs = []
    for i in range(0, len(df)):
        if i == 100:
            print("100 done")
        if i == 1000:
            print("1000 done")
        if i == 5000:
            print("5000 done")
        if i == 7000:
            print("7000 done")
        if i == 9000:
            print("9000 done")

        prompt = df['prompt'][i] 
        new_prompt = prompt.replace("Answer with yes/no/can't tell", "Answer first only with yes or no, then provide your reasoning")
        output = moderate([
        {"role": "user", "content": new_prompt},
        ])
        guard_outputs.append(output)
    return guard_outputs

llama_guard = run_llama_guard()
data = {"safe label" : llama_guard}
df_guard = pd.DataFrame(data)
#save df as necessary
print('df done')

#### Mistral Moderation API ####
client = Mistral(api_key=api_key)

def moderate_conversation(client: Mistral, user_prompt: str) -> dict:
    """
    Moderate the conversation using the Mistral moderation API.

    Args:
        client (Mistral): The Mistral client instance.
        user_prompt (str): The user prompt.
        response (str): The assistant response.

    Returns:
        dict: The moderation results.
    """
    response = client.classifiers.moderate_chat(
        model="mistral-moderation-2411",
        inputs=[
            {"role": "user", "content": user_prompt},

        ],
    )
    return response

outputs = []

for i in range(0, len(df)):
  prompt = df['prompt'][i]
  if multiples.count(i/100) > 0:
    print('sleeping')
    time.sleep(60)
    print('continuing at', i)
  new_prompt = prompt.replace("Answer with yes/no/can't tell", "Answer first only with yes or no, then provide your reasoning")
  outputs.append(moderate_conversation(client, new_prompt))