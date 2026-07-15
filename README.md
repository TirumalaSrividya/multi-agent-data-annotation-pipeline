# Multi-Agent Data Annotation & Active Learning Pipeline

## Overview

A Multi-Agent AI pipeline that automatically annotates unlabelled text data using an LLM, validates annotation quality, and trains multiple machine learning models to select the best-performing classifier.

The system consists of two pipelines:

1. Annotation Pipeline – Selects novel samples, annotates them using an LLM, and validates low-confidence predictions.
2. Training Pipeline – Trains multiple candidate models, evaluates them, and selects the best model based on evaluation metrics.


## Architecture

``` 
                        +----------------------+
                        |    Orchestrator      |
                        +----------+-----------+
                                   |
          -------------------------------------------------
          |                                               |
          ▼                                               ▼

   Annotation Pipeline                         Training Pipeline

   +--------------+                             +---------------+
   | Sampler      |                             | Trainer Agent |
   | Agent        |                             +-------+-------+
   +------+-------+                                     |
          |                                             |
          ▼                                             ▼
   +--------------+                          Train KNN / LSTM / RNN
   | Annotator    |                                     |
   | Agent        |                                     ▼
   +------+-------+                           Evaluate Models
          |                                             |
          ▼                                             ▼
   +--------------+                          Select Best Model
   | Quality      |                                     |
   | Assessor     |                                     ▼
   +------+-------+                          Generate Training Report
          |
          ▼
 Quality-Assessed Labelled Dataset
```


## Folder Structure

```
multi_agent_annotation_pipeline/

├── artifacts/                  # Generated outputs
│
├── config/
│   └── config.yaml             # Application configuration
│
├── data/
│   └── sample_news.csv         # Input dataset
│
├── scripts/
│   ├── generate_sample_data.py
│   └── run_pipeline.py
│
├── src/
│   ├── agents/
│   │   ├── annotator_agent.py
│   │   ├── quality_assessor_agent.py
│   │   ├── sampler_agent.py
│   │   ├── trainer_agent.py
│   │   └── base.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   └── embeddings.py
│   │
│   ├── ml/
│   │   ├── models.py
│   │   ├── train.py
│   │   └── metrics.py
│   │
│   ├── utils/
│   │
│   ├── config.py
│   ├── llm_client.py
│   ├── logging_setup.py
│   ├── orchestrator.py
│   └── schemas.py
│
├── tests/
│   ├── test_annotator_agent.py
│   ├── test_quality_assessor_agent.py
│   ├── test_sampler_agent.py
│   ├── test_trainer_agent.py
│   ├── test_orchestrator.py
│   ├── test_metrics.py
│   └── test_token_budget.py
│
├── requirements.txt
└── README.md

```

## Setup

## clone repository

git clone
```
https://github.com/TirumalaSrividya/multi-agent-data-annotation-pipeline 
```
cd Agentic_Annotation Pipeline
```

## Install Dependencies
```
pip install -r requirements.txt
pip install pytest
```

##  Start Ollama

bash 
```
ollama pull llama3.2 
ollama serve

```

## Run Pipeline

```bash
python scripts/run_pipeline.py
```

## Run Tests

```bash
python -m pytest
```
