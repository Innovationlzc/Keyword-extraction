# **Keyword-Extraction**

**Parameter-Efficient Keyphrase Generation via Low-Rank Adaptation on small LLMs**

------

## **Introduction**

This repository fine-tunes a T5-small & DistillBert-inspec backbone with **LoRA (Low-Rank Adaptation)** for automatic keyword extraction.  
Key highlights:

- **PEFT**-style training—updates < 1 % of parameters.  
- Supports popular datasets: **SemEval-ScienceIE 2017, Krapivin-2009, Inspec**, plus any custom JSON.  
- Three main scripts  
  - `finetune_lora_T5.py` → training  
  - `prediction_T5.py`  → inference / generation  
  - `evaluate.py`    → Precision / Recall / F1@K scorer  
- Baseline runners for TF-IDF, YAKE, TextRank, KeyBERT (see `demo/`).

------

## **Usage**

### **1. Setup**

```bash
git clone https://github.com/Innovationlzc/Keyword-extraction.git
cd Keyword-extraction

# create environment
conda create -n keyword-lora python=3.10
conda activate keyword-lora
pip install -r requirements.txt          # transformers, peft, datasets, …
