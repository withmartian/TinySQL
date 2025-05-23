﻿# TinySQL

# Introduction
How does a language model use this prompt:
```
### Instruction: How far is each destination?
### Context: CREATE TABLE locations ( place TEXT, distance INTEGER, population FLOAT, transport_mode TEXT) 
### Response:
```
To generate this output?
```
SELECT distance from locations
```
The "TinySQL" project investigates models that perform "english text to sql commands" predictions.
This repository contains project artefacts.

# Approach
This project:
- defines 3 distinct subsets of SQL (called "command sets') of differing complexity
- generates training data for each command set 
- refines small (33M params) and large (0.5B and 1B param) models on each command set giving models high prediction accuracy 
- investigates the algorithm used by the models using Mechanistic Interpretability (MI) techniques

# Resources
Resources associated with this project include:
- Command set definitions https://docs.google.com/document/d/1HZMqWJA5qw8TFhyk8j3WB573ec-8bKMdrE4TnV4-qgc
- Training data and trained modesl on huggingface https://huggingface.co/collections/withmartian/tinysql-6760e92748b63fa56a6ffc9f
- Notebooks that do Mechanistic Interpretability (MI) investigations on the trained models. Details below.

## Folders, Files and Classes 
This library contains files:
- **TinySQL:** Python library code imported into the notebooks:
  - load_data: Load the supported base models openai-community/gpt2, Qwen/Qwen2.5-0.5B-Instruct, withmartian/Llama-3.2-1B-Instruct and the trained model derived from them.
  - training_data: Generate training data associated with the command sets. Also evaluate (score) the accuracy of model predictions.
  - gretel_dd: Generate training data associated with the command sets using the [Gretel AI](https://gretel.ai/) synthetic data generation tools. 
  - finetune: Using the training data, finetune the base models to give the trained models.
  - corrupt_data: Generate pairs of clean and corrupted data for use in MI experiments
- **notebooks**: Most of these Jupyter notebooks can be run on any supported (base or trained) model with any command set.
  - tinysql_generate: Shows that, given generated data, models can predict answer accurately  
  - tinysql_activation_patching_json: Activation patching experiments
  - tinysql_attribution_patching: Attribution patching experiments
  - tinysql_eap_json: Example of attribution patching using nnsight library
  - tinysql_m1_useful_nodes: Identify the model nodes that are necessary for accurate predictions in BM1.  
  - tinysql_semantic_map: Investigate how models handle semantic prompts (like the example at top) 
  - sae_training: Train Sparse AutoEncoders against our refined models.
  - sae_interp: Interpret Sparse AutoEncoders results.
  - analyze_tiny_stories: Generate statistics from our wandb data  
**test**: unit tests

## Base Models
These base models were fine-tuned to give the TinySQL models:
- TinyStories-Instruct-2Layers-33M
- Qwen2.5-0.5B-Instruct
- Llama-3.2-1B-Instruct 

## Installation
From source

```bash
git clone https://github.com/withmartian/TinySQL.git
cd TinySQL
pip install .
```

## Thanks
We are grateful to: 
- [WithMartian](https://withmartian.com/) for supporting this project financially.
- [Gretel AI](https://gretel.ai/) for the use of their excellent synthetic data generation tools. 
- The [nnsight](https://nnsight.net/) authors for their (low level) library that support allow experiments to be written against toy and large language models in a reusable way.
- The [QuantaMechInterp](https://github.com/PhilipQuirke/quanta_mech_interp) library authors for their tools that help investigate and visualize model algorithms using any MI technique.
