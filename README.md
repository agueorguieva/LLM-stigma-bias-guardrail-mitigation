#Identifying Features Associated with Bias Against 93 Stigmatized Groups in Language Models and Guardrail Model Safety Mitigation

Authors: Anna-Maria Gueorguieva, Aylin Caliskan

Corresponding author for paper and github: Anna-Maria Gueorguieva, agueorg@uw.edu

This README file contains an overview of the code and data for "Identifying Features Associated with Bias Against 93 Stigmatized Groups in Language Models and Guardrail Model Safety Mitigation" published in this github repository.


## Data

results-from-pachankis-all.csv contains the outputs from LLama, Mistral, and Granite on rating each of the 93 stigmas across 6 features. It is obtained using run\_pachankis\_on\_LLMs.py in the code section and analyzed in analysis-and-visualizations.R

SSQA-results.csv contain each models output to the given questions in SocialStigamQA benchmark and compares that answer to SocialStigmaQA to determine if a model outputted a biased answer; obtained using SSQA\_benchmarking.py 

The folder "raw-guardrail-model-results" contains guardian-bias-detection-results.csv, guardian-harm-detection-results.csv, llama\_guard\_all.csv, and mistral-moderation-detection-results.csv ; all of these contain data to their respective guardrail's performance on SocialStigmaQA and are obtained by running run-guardrails.py

The remaining csv's in the data folder are processed based on the previously mentioned csv's. They are processed for easier analysis and graphing using R and their processing is done in data-processing.ipynb

SocialStigmaQA dataset accessed using pd.read\_csv("hf://datasets/ibm/SocialStigmaQA/yes\_no/full\_dataset\_yes\_no.csv") 

## Code

run\_pachankis\_on\_LLMs.py and helper\_functions.py is used in Experiment 1 (Section 6.1) to create the data results-from-pachankis-all.csv. This results of this data are reported in Section 7.1. 

SSQA\_benchmarking.py is used in Experiment 2 (Section 6.2) to benchmark the language models on SocialStigmaQA; the data is saved in SSQA-results.csv with results reported in 7.2. 

run-guardrails.py is used in Experiment 3 (Section 6.3) to obtain success of guardrail models in identifying harm in inputs. The data is processed in data-processing.ipynb and the final analysis is conducted on data files: llama\_and\_llama\_guard.csv, granite\_and\_granite\_guardian.csv, mistral\_and\_mistral\_moderation.csv and reported in section 7.3. 

Data not directly obtained from LLMs or their guardrails using relevant .py files was processed within data-processing.ipynb to get formatting that was easier for analysis and visualization in R.

All analysis and visualizations are conducted on analysis-and-visualizations.R and contain references to where in the paper the results are reported. 
