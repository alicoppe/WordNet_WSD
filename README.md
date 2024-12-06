# WordNet_WSD

# Word Sense Disambiguation Project

This repository contains the implementation and evaluation of various methods for Word Sense Disambiguation (WSD) as part of COMP 550 - Programming Assignment 2. The project aims to determine the intended sense of ambiguous words in context using diverse approaches and datasets.

## Overview

Word Sense Disambiguation (WSD) addresses the challenge of assigning the correct meaning to words based on their context. This project evaluates several methods using the **SemEval 2013 Shared Task #12** dataset and **WordNet v3.0**. The implemented methods are:

- **Baseline:** Assigns the most frequent sense of each word as per WordNet.
- **Lesk's Algorithm:** Matches the context of the target word with WordNet definitions.
- **BERT-based Sentence-Level Embeddings:** Uses pretrained BERT to compare sentence embeddings with WordNet definitions and examples.
- **Google Gemini 1.5 LLM Prompting:** Leverages a Large Language Model for direct WSD predictions.

## Results Summary

| Method                     | Development Set Accuracy | Test Set Accuracy |
|----------------------------|--------------------------|--------------------|
| Baseline                  | 65.46%                  | 62.14%            |
| Lesk's Algorithm          | 35.57%                  | 33.79%            |
| BERT Sentence Embeddings  | 52.06%                  | 46.90%            |
| Google Gemini 1.5         | 64.95%                  | 73.24%            |

The Google Gemini model achieved the highest test accuracy, demonstrating the potential of LLMs for WSD tasks.

## Key Features

1. **Preprocessing:**
   - Text lowercasing, punctuation removal, lemmatization, and stopword filtering.
   - Special handling for acronyms and hyphenated words.

2. **Lesk's Algorithm:**
   - Implementation via NLTK's `lesk` function with a focus on contextual overlap with WordNet glosses.

3. **BERT Sentence Embeddings:**
   - Sentence-level embeddings from a pretrained `bert-base-uncased` model.
   - Comparison using cosine similarity and Euclidean distance.

4. **LLM Prompting:**
   - Google Gemini 1.5 API with structured prompts to predict WordNet synsets.
   - Analysis of the impact of prompt length on accuracy.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/wsd-project.git
   cd wsd-project
