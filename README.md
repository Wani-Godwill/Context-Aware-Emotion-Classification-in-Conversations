
#  Context-Aware Emotion Classification in Conversations

> **A Comparative NLP Study Using Classical Machine Learning, Sentence-BERT, and Fine-Tuned Transformers**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-green.svg)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

# Overview

This project investigates **emotion classification in conversations** and, in particular, whether **conversational context improves the prediction of emotions from individual utterances**.

The project develops and compares multiple approaches:

* Classical machine learning baseline
* Sentence-BERT + Logistic Regression
* Fine-tuned DistilRoBERTa transformer

A central contribution of the project is the evaluation of different context conditions:

* No context
* Previous three conversational turns
* Shuffled context
* Unrelated context

This experimental design makes it possible to investigate whether performance improvements are caused by **genuine conversational context** rather than simply by providing the model with additional text.

---

# Project Objectives

The main objectives of the project are to:

* Prepare and process a conversational emotion dataset.
* Explore the characteristics and distribution of the data.
* Develop a classical machine learning baseline.
* Evaluate Sentence-BERT as a semantic feature extractor.
* Fine-tune a transformer model for emotion classification.
* Compare context-aware and context-free models.
* Investigate the usefulness of genuine conversational context.
* Compare Sentence-BERT and transformer approaches.
* Evaluate the final models on a held-out test set.
* Perform detailed quantitative and qualitative error analysis.

---

# Project Pipeline

The complete workflow consists of:

```text
Original Dataset
      │
      ▼
Data Preparation
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Classical Baseline
      │
      ▼
Sentence-BERT Experiments
      │
      ▼
Fine-Tuned Transformer Experiments
      │
      ▼
Validation-Based Model Selection
      │
      ▼
Held-Out Test Evaluation
      │
      ▼
Error and Context Analysis
```

---

# Dataset and Context Preparation

The original conversational emotion dataset is processed and divided into:

* **Training set**
* **Validation set**
* **Held-out test set**

The processed dataset is used throughout the project to ensure that all models are evaluated using the same underlying examples.

## Context Representations

Four different input representations are created.

### No Context

The model receives only the current utterance.

```text
Current utterance
```

---

### Previous 3 Turns

The model receives the current utterance together with the previous three conversational turns.

```text
Previous Turn 3
      ↓
Previous Turn 2
      ↓
Previous Turn 1
      ↓
Current Utterance
```

This represents the **genuine conversational context**.

---

### Shuffled Context

The context is provided in an incorrect or shuffled order.

This experiment investigates whether the model benefits from the **correct order of conversational information**.

---

### Unrelated Context

The model receives context from unrelated examples or conversations.

This provides a control condition to test whether improvements result from meaningful context or simply from additional text.

---

# Exploratory Data Analysis

The dataset is analysed before model development.

The analysis includes:

* Emotion class distribution
* Class imbalance
* Number of dialogues
* Dialogue length
* Utterance length
* Missing values
* Conversational structure
* Examples of emotion-labelled utterances

Because the emotion classes are imbalanced, accuracy alone is not sufficient for model evaluation.

---

# Models

## Classical Machine Learning Baseline

A traditional machine-learning baseline is developed to provide a reference point for the more advanced models.

The baseline follows the general structure:

```text
Text
 │
 ▼
Text Representation
 │
 ▼
Machine Learning Classifier
 │
 ▼
Emotion Prediction
```

The purpose of the baseline is to determine whether more advanced language models provide meaningful improvements.

---

# Sentence-BERT + Logistic Regression

Sentence-BERT is used as a **frozen semantic feature extractor**.

The workflow is:

```text
Text / Conversational Context
            │
            ▼
      Sentence-BERT
            │
            ▼
     Fixed Embeddings
            │
            ▼
   Logistic Regression
            │
            ▼
    Emotion Prediction
```

The Sentence-BERT encoder is not fine-tuned during classification.

The following context experiments are compared:

* No context
* Previous 3 turns
* Shuffled context
* Unrelated context

Logistic Regression hyperparameters and class-weighting strategies are selected using **validation Macro-F1**.

---

# Fine-Tuned Transformer

The deep learning approach uses a **fine-tuned DistilRoBERTa transformer**.

Unlike Sentence-BERT, the transformer is fine-tuned directly for the emotion classification task.

The architecture is:

```text
Input Text
    │
    ▼
Tokenizer
    │
    ▼
DistilRoBERTa
    │
    ▼
Classification Head
    │
    ▼
Emotion Prediction
```
