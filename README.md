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

### **Setup**

```bash
git clone https://github.com/Innovationlzc/Keyword-extraction.git
cd Keyword-extraction

# create environment
conda create -n keyword-lora python=3.10
conda activate keyword-lora
pip install -r requirements.txt          # transformers, peft, datasets, …
```


## **Key Functions**

| File / Script         | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `finetune_lora_T5.py` | Fine-tunes a T5-small model using LoRA (Low-Rank Adaptation). Supports full training config customization including LoRA rank, alpha, dropout, learning rate, and batch size. |
| `prediction_T5.py`    | Loads the base T5-small model and LoRA adapter to generate comma-separated keywords for each input text.|
| `evaluate.py`         | Computes Precision@K, Recall@K, and F1@K for predicted vs. ground truth keywords. Supports both exact matching and semantic similarity (via Sentence-BERT). |
| `baseline/`           | |
| `data/*`              | Preprocessed Train and Test data. |
| `demo/*`              | Examples of outputs. |
| `requirements.txt`    | Dependency Library|



