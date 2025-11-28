
# Comparative Analysis of 47 Context-Based Question Answering Models Across 8 Diverse Datasets  
**Historical Benchmark – August to September 2023**  
**Repository Status: ARCHIVED (November 2025)**  

![Python](https://img.shields.io/badge/Python-3.9+-blue)  
![Transformers](https://img.shields.io/badge/Transformers-4.x-orange)  
![License](https://img.shields.io/badge/License-MIT-green)

## Overview & Historical Context

This repository contains the complete code, results, figures, and LaTeX source of a large-scale benchmarking study conducted **August – September 2023**.

At that time:
- The public LLM boom had only just begun (GPT-4: March 2023, Llama-2: July 2023)
- Most research labs could **not** run 7B+ models locally
- There was a real need for **lightweight, fully offline, privacy-preserving** extractive QA models that could be embedded in local pipelines

The main practical goal was to answer the question:  
> “Which <400 MB Hugging Face model extracts short factual answers (≤5 words) from full-text scientific papers most accurately and fastest on a normal laptop or small server?”

This repository is the answer to that 2023 question — and is now preserved purely as a **historical artifact**.

**2025 perspective**: Modern open-weight LLMs (Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, Phi-3.5, etc.) solve the same task at 80–95 % accuracy in zero-shot or 4-bit mode. The 2023 extractive QA models are obsolete for practical use.

## Key Results & Findings (2023)

### Overall Top 5 Models
| Rank | Model Name                                                | Size (MB) | Training Dataset          | Overall Accuracy |
|------|-----------------------------------------------------------|-----------|---------------------------|------------------|
| 1    | `ahotrod/electra_large_discriminator_squad2_512`         | 319       | SQuAD v2                  | **43.27%**      |
| 2    | `bert-large-uncased-whole-word-masking-finetuned-squad`  | 320       | BookCorpus + Wikipedia    | 42.26%          |
| 3    | `Palak/microsoft_deberta-large_squad`                    | 387       | SQuAD v1                  | 42.21%          |
| 4    | `bhadresh-savani/electra-base-squad2`                    | 104       | SQuAD v2                  | 40.70%          |
| 5    | `twmkn9/albert-base-v2-squad2`                           | 12        | SQuAD v2                  | 38.42%          |

### Best Model per Dataset (2023)
| Dataset                  | Best Model                                                      | Accuracy  |
|--------------------------|------------------------------------------------------------------|-----------|
| biomedical_cpgQA         | ahotrod/electra_large_discriminator_squad2_512                  | **96.45%**|
| IELTS                    | bert-large-uncased-whole-word-masking-finetuned-squad           | 82.00%   |
| bioasq10b-factoid        | ahotrod/electra_large_discriminator_squad2_512                  | 65.92%   |
| Question Answer Dataset  | ahotrod/electra_large_discriminator_squad2_512                 | 41.60%   |
| JournalQA                | Palak/microsoft_deberta-large_squad                              | 31.00%   |
| ScienceQA                | twmkn9/albert-base-v2-squad2                                   | 24.60%   |
| QuAC                     | ahotrod/electra_large_discriminator_squad2_512                  | 11.13%   |
| atlas-math-sets          | bhadresh-savani/electra-base-squad2                             | 3.72%    |

### Important Patterns Discovered
- Top models were almost exclusively fine-tuned on **SQuAD v1/v2**
- **Model size vs accuracy**: –8 % correlation → bigger ≠ better
- **Execution time**: +51 % correlation with model size; context length also major factor
- **Answer length effect**: accuracy drops sharply as answer length increases (1→5 words)
- **Question-type routing works**: in JournalQA, different models were best for “Dataset”, “Performance”, “Number”, “Location”, etc.
- **Ensembles**: genetic-algorithm combinations gave **no statistically significant gain**

## Datasets Used
| Dataset                  | Questions | Domain                     | Avg Context Length | Answer ≤5 words |
|--------------------------|-----------|----------------------------|-------------------|-----------------|
| atlas-math-sets          | 10,000   | Mathematics                 | 3                 | Yes             |
| bioasq10b-factoid        | 983       | Biomedical                  | 351               | Yes             |
| biomedical_cpgQA         | 535       | Clinical Guidelines         | 161               | Yes             |
| IELTS                    | 50        | Reading Comprehension       | 907               | Yes             |
| JournalQA                | 273       | Scientific Articles         | 5,568            | Yes             |
| QuAC                     | 618       | Conversational QA           | 84                | Yes             |
| ScienceQA                | 1,378     | General Science             | 147               | Yes             |
| Question Answer Dataset     | 2,037     | Wikipedia-based             | 5,227            | Yes             |

## Repository Structure
```
├── code/          # All scripts (evaluation, analysis, plots)
├── datasets/      # Processed CSVs (merge with OSF download)
├── results/       # Raw predictions & metrics
├── figures/       # Heatmaps, plots, flowchart
├── paper/         # Full LaTeX source (unpublished)
├── Flowchart.png  # Visual workflow
├── requirements.txt
└── README.md
```

## Data Download (Required)
Processed datasets are hosted on OSF:  
https://osf.io/rd3x2/?view_only=7f9bd36e0d974268b610419e8241dc8a  

Download → unzip → merge with the `datasets/` folder.
 
## Citation (historical reference only)
```bibtex
@misc{muneeb2023cbqa,
  author       = {Muhammad Muneeb and David B. Ascher and Ahsan Baidar Bakht},
  title        = {Comparative Analysis of 47 Context-Based Question Answering Models Across 8 Diverse Datasets: A 2023 Historical Benchmark},
  year         = {2023},
  month        = {sep},
  howpublished = {\url{https://github.com/MuhammadMuneeb007/Comparative-Analysis-of-47-Context-Based-Question-Answer-Models-Across-8-Diverse-Datasets}},
  note         = {Archived — superseded by modern LLMs}
}
```

## Authors
- **Muhammad Muneeb** – University of Queensland & Baker Institute  
- **David B. Ascher** (corresponding) – d.ascher@uq.edu.au  
 

**This was excellent, thorough work — for 2023.**  
In 2025 it is proudly archived as a clean snapshot of the pre-LLM extractive QA research.
 


Just copy-paste the entire text above as your new `README.md`, commit, and archive the repository. You’re all set.
