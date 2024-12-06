import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import nltk
import string
from nltk.stem import WordNetLemmatizer
from loader import load_instances, load_key
import re
from nltk.corpus import stopwords
from nltk.wsd import lesk
import random
from scipy.spatial.distance import cosine, euclidean
from nltk.corpus import wordnet as wn
import pandas as pd

from transformers import BertTokenizer, BertModel

import sys
import time

import google.generativeai as genai
import os
from google.cloud import aiplatform
from vertexai.preview.tokenization import get_tokenizer_for_model

path_to_key = '/Users/aidanlicoppe/Documents/Code/keys'

# Add the full file path of the directory containing api_keys.py
sys.path.append(path_to_key)

from api_keys import google_api_key


# Load instances and key data in another cell
data_f = 'multilingual-all-words.en.xml'  
key_f = 'wordnet.en.key'

dev_instances, test_instances = load_instances(data_f)
dev_key, test_key = load_key(key_f)

dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}


lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(word):
    # Remove punctuation
    word = word.translate(str.maketrans('', '', string.punctuation))
    # Lemmatize
    return lemmatizer.lemmatize(word)

def create_dataframe(instances, key_dict):
    data = []
    for instance_id, instance in instances.items():
        # Decode lemma and context if in byte format
        lemma = instance.lemma.decode('utf-8') if isinstance(instance.lemma, bytes) else instance.lemma
        context = [word.decode('utf-8') if isinstance(word, bytes) else word for word in instance.context]

        # Retrieve the sense key(s) from key_dict, or None if not found
        sense_key = key_dict.get(instance_id, [None])
        
        # Append the processed data
        data.append({
            'Instance ID': instance_id,             
            'Lemma': lemma,                         
            'Original Context': context,
            'Combined Context': ' '.join(context),          
            'Index': instance.index,               
            'Sense Key': sense_key                  
        })
    return pd.DataFrame(data)


dev_df = create_dataframe(dev_instances, dev_key)
test_df = create_dataframe(test_instances, test_key)


stop_words = set(stopwords.words("english"))

def preprocess_context(context):
    processed_context = []
    for word in context:
        # Convert to lowercase
        word = word.lower()

        # Handle "@card@" tokens and numeric values by replacing with "NUM"
        if word == "@card@" or re.fullmatch(r'\d+', word):
            processed_context.append("NUM")
            continue
        
        # Preserve periods within abbreviations and replace with underscores (e.g., "u.n." -> "u_n")
        word = re.sub(r'\b(\w\.)+', lambda match: match.group(0).replace('.', '_'), word)
        
        # Split hyphenated compound words (e.g., "u_n-sponsored" -> ["u_n", "sponsored"])
        parts = re.split(r'-(?=\w)', word)
        
        # Process each part separately
        for part in parts:
            # Remove isolated punctuation from each part
            part = part.strip(string.punctuation)
            
            # Lemmatize, remove stop words, and add to processed context if not empty
            
            # if part and part not in stop_words:
            #     processed_context.append(lemmatizer.lemmatize(part))
                
            if part:
                processed_context.append(lemmatizer.lemmatize(part))
    
    return processed_context

# Example application
dev_df['Modified Context'] = dev_df['Original Context'].apply(preprocess_context)
test_df['Modified Context'] = test_df['Original Context'].apply(preprocess_context)

dev_df['Combined Modified Context'] = dev_df['Modified Context'].apply(lambda x: ' '.join(x))
test_df['Combined Modified Context'] = test_df['Modified Context'].apply(lambda x: ' '.join(x))


def calculate_accuracy(predicted_synsets, actual_synsets, get_name=False):

    correct_predictions = 0
    correct_indices = []
    incorrect_indices = []
    
    for i, (predicted, actual) in enumerate(zip(predicted_synsets, actual_synsets)):
        if get_name:
            actual = actual.name()
        
        if predicted == actual:
            correct_predictions += 1
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)

    # Calculate accuracy
    accuracy = correct_predictions / len(actual_synsets)
    return accuracy, correct_indices, incorrect_indices


df = test_df

actual_synsets = [wn.lemma_from_key(row['Sense Key'][0]).synset() for _, row in df.iterrows()]

# Generate predicted labels using Lesk
lesk_predictions = [lesk(row['Modified Context'], row['Lemma']) for _, row in df.iterrows()]

baseline_predictions = [wn.synsets(row['Lemma'])[0] if wn.synsets(row['Lemma']) else None for _, row in df.iterrows()]

# Calculate accuracy and get indices for correct and incorrect predictions
lesk_accuracy, lesk_correct_indices, lesk_incorrect_indices = calculate_accuracy(lesk_predictions, actual_synsets)

baseline_accuracy, baseline_correct_indices, baseline_incorrect_indices = calculate_accuracy(baseline_predictions, actual_synsets)

# Print rounded accuracy values
print('--' * 20)
print(f"Lesk Accuracy: {lesk_accuracy*100:.2f}%")
print(f"Baseline Accuracy: {baseline_accuracy*100:.2f}%")
print('--' * 20, '\n')

def lesk_with_full_debug(context, word, actual_synset):
    print(f"Ambiguous word: {word}")
    print(f"\nContext (lemmatized): {context}")
    
    # Run NLTK's Lesk algorithm and get the predicted synset
    predicted_synset = lesk(context, word)
    print(f"\nPredicted Synset: {predicted_synset}")
    print(f"Definition of Predicted Synset: {predicted_synset.definition() if predicted_synset else 'None'}")
    
    # Print the actual synset for comparison
    print(f"Actual Synset: {actual_synset}")
    print(f"Definition of Actual Synset: {actual_synset.definition() if actual_synset else 'None'}")
    
    print("\nNumber of possible synsets:", len(wn.synsets(word)))
    
    print("\nAll possible synsets and their overlap scores:\n")

    # Calculate and display overlap for each synset of the ambiguous word
    lemmatized_context = set(context)
    for synset in wn.synsets(word):
        # Gloss, examples, and hypernyms words
        gloss_words = set(synset.definition().split())
        
        overlap_words = lemmatized_context.intersection(gloss_words)
        overlap_score = len(overlap_words)
        
        # Display details for this synset
        print(f"Synset: {synset}")
        print(f"Definition: {synset.definition()}")
        print(f"Overlap Words: {overlap_words}")
        print(f"Overlap Score: {overlap_score}\n")

    return predicted_synset

correct_samples = random.sample(lesk_correct_indices, 1) 
incorrect_samples = random.sample(lesk_incorrect_indices, 1)  

# Display Lesk's computation for correct predictions
print("\n----------------------------------------- Correct Prediction Debug -----------------------------------------")
for idx in correct_samples:
    row = df.iloc[idx]
    actual_synset = actual_synsets[idx]
    print(f"\nInstance ID: {row['Instance ID']}")
    lesk_with_full_debug(row['Modified Context'], row['Lemma'], actual_synset)

# Display Lesk's computation for incorrect predictions
print("\n----------------------------------------- Incorrect Prediction Debug -----------------------------------------")
for idx in incorrect_samples:
    row = df.iloc[idx]
    actual_synset = actual_synsets[idx]
    print(f"\nInstance ID: {row['Instance ID']}")
    lesk_with_full_debug(row['Modified Context'], row['Lemma'], actual_synset)
    
    

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Function that takes in a sentence in the form of a string and returns it lemmatized with stop words and punctuation removed
def process_sentence(sentence):
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    
    # Lemmatize each token and remove stop words and punctuation
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation]
    
    # Join the lemmatized tokens into a single sentence
    lemmatized_sentence = ' '.join(lemmatized_tokens)
    
    return lemmatized_sentence


def get_sentence_embedding(sentence):
    # Tokenize the sentence and get the embeddings for all tokens
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    # Take the mean of the token embeddings to represent the sentence
    sentence_embedding = last_hidden_states.mean(dim=1).squeeze()
    return sentence_embedding.detach().numpy()

def get_synset_embeddings(target_word, comparison_type='definition'):
    # Create a dictionary to store sentence embeddings for each synset example
    synset_embeddings = {}
    for synset in wn.synsets(target_word):
        # Use the first example sentence of the synset if available
        if comparison_type == 'examples':
            if synset.examples():
                example_sentence = synset.examples()[0]
                # print(f"Example sentence for synset '{synset}': {example_sentence}")
                # Get the sentence-level embedding
                synset_embedding = get_sentence_embedding(example_sentence)
                synset_embeddings[synset] = synset_embedding
        if comparison_type == 'definition':
            if synset.definition():
                definition_sentence = synset.definition()
                # definition_sentence = process_sentence(definition_sentence)
                # Get the sentence-level embedding
                synset_embedding = get_sentence_embedding(definition_sentence)
                synset_embeddings[synset] = synset_embedding
    return synset_embeddings


def predict_synset_BERT(sentences, target_word, metric='cosine', comparison_type='definition'):
    """
    Predict the most likely synset for each sentence based on the target word's context.

    Parameters:
    - sentences: list of sentences containing the target word.
    - target_word: the word to disambiguate.
    - metric: the type of distance metric to use ('cosine' for cosine similarity, 'euclidean' for Euclidean distance).

    Returns:
    - predicted_synsets: a list of the predicted synsets for each sentence.
    """
    # Get synset sentence embeddings for the target word
    synset_embeddings = get_synset_embeddings(target_word, comparison_type=comparison_type)
    
    predicted_synsets = []
    
    for sentence in sentences:
        # sentence = process_sentence(sentence)
        # Get the sentence-level embedding for the input sentence
        sentence_embedding = get_sentence_embedding(sentence)
        
        # Initialize variables to track the best matching synset
        best_synset = None
        if metric == 'cosine':
            best_score = -1
        elif metric == 'euclidean':
            best_score = float('inf')
        
        # Compare with each synset embedding based on the chosen metric
        for synset, synset_embedding in synset_embeddings.items():
            if synset_embedding is not None:
                # Calculate similarity or distance based on the metric
                if metric == 'cosine':
                    score = 1 - cosine(sentence_embedding, synset_embedding)  # Higher is better for cosine similarity
                elif metric == 'euclidean':
                    score = euclidean(sentence_embedding, synset_embedding)   # Lower is better for Euclidean distance
                else:
                    raise ValueError("Unsupported metric. Choose either 'cosine' or 'euclidean'.")
                
                # Determine best score and synset based on the metric
                if ((metric == 'cosine' and score > best_score) or (metric == 'euclidean' and score < best_score)):
                    best_score = score
                    best_synset = synset
                    
        if best_synset is None:
            best_synset = wn.synsets(target_word)[0]
        
        # Append the best matching synset for the sentence
        predicted_synsets.append(best_synset)
    
    return predicted_synsets


df = dev_df

# Extract sentences and lemmas
sentences = df['Combined Context'].tolist()
lemmas = df['Lemma'].tolist()

# Initialize an array to store the predicted synsets
predicted_synsets_cosine = []
predicted_synsets_euclidean = []

# Run the prediction for each sentence and lemma pair
for sentence, lemma in zip(sentences, lemmas):
    # Run predict_synset_BERT on each sentence-lemma pair individually
    synset_prediction_cosine = predict_synset_BERT([sentence], lemma, metric='cosine')
    # Extract the predicted synset from the list and append
    predicted_synsets_cosine.append(synset_prediction_cosine[0])
    
    
actual_synsets = [wn.lemma_from_key(row['Sense Key'][0]).synset() for _, row in df.iterrows()]

acc_cos, correct_cos, incorrect_cos = calculate_accuracy(predicted_synsets_cosine, actual_synsets)


print("\n----------------------------------------- Cosine Similarity -----------------------------------------")
print(f"Accuracy for cosine similarity: {acc_cos*100:.2f}%")
print('--' * 20, '\n')

def BERT_with_full_debug(context, word, actual_synset):
    print(f"Ambiguous word: {word}")
    print(f"\nContext (lemmatized): {context}")
    
    # Run NLTK's Lesk algorithm and get the predicted synset
    predicted_synset = predict_synset_BERT([context], word, metric='cosine')
    predicted_synset = predicted_synset[0]
    print(f"\nPredicted Synset: {predicted_synset}")
    print(f"Definition of Predicted Synset: {predicted_synset.definition() if predicted_synset else 'None'}")
    
    # Print the actual synset for comparison
    print(f"Actual Synset: {actual_synset}")
    print(f"Definition of Actual Synset: {actual_synset.definition() if actual_synset else 'None'}")
    
    print("\nNumber of possible synsets:", len(wn.synsets(word)))
    
    print("\nAll possible synsets and their cosine similarity:\n")

    for synset in wn.synsets(word):
        # Print definition sentence
        definition_sentence = synset.definition()
        print(f"Definition sentence for synset '{synset}': {definition_sentence}")
        
        # Get the sentence-level embedding
        synset_embedding = get_sentence_embedding(definition_sentence)
        
        # Calculate cosine similarity
        similarity = 1 - cosine(get_sentence_embedding(context), synset_embedding)
        
        print(f"Synset: {synset}, Cosine Similarity: {similarity}\n")

    return predicted_synset

correct_samples = random.sample(correct_cos, 1) 
incorrect_samples = random.sample(incorrect_cos, 1)  

print("\n----------------------------------------- Correct Prediction Debug -----------------------------------------")
for idx in correct_samples:
    row = df.iloc[idx]
    actual_synset = actual_synsets[idx]
    print(f"\nInstance ID: {row['Instance ID']}")
    BERT_with_full_debug(row['Combined Context'], row['Lemma'], actual_synset)
    
print("\n----------------------------------------- Incorrect Prediction Debug -----------------------------------------")
for idx in incorrect_samples:
    row = df.iloc[idx]
    actual_synset = actual_synsets[idx]
    print(f"\nInstance ID: {row['Instance ID']}")
    BERT_with_full_debug(row['Combined Context'], row['Lemma'], actual_synset)
    




genai.configure(api_key=google_api_key)

model = genai.GenerativeModel("gemini-1.5-flash")

tokenizer = get_tokenizer_for_model("gemini-1.5-flash-002")


def get_synset_codes(target_word):
    """
    Retrieve the synset codes and definitions for the target word.
    """
    synsets = wn.synsets(target_word)
    synset_info = [(synset.name(), synset.definition()) for synset in synsets]
    return synset_info

def create_prompt_batches(sentences, lemmas, words_per_prompt=50):
    # Words per prompt specifies the number of target lemmas per prompt
    prompts = []
    current_prompt = "For each sentence below, choose the correct WordNet synset code for the target word in context.\n\n"
    
    total_length = len(sentences)
    lemmas_in_prompt = []

    current_lemma_list = []
    for i, (sentence, lemma) in enumerate(zip(sentences, lemmas)):
        synset_info = get_synset_codes(lemma)
        synset_descriptions = "\n".join([f"{code}: {definition}" for code, definition in synset_info])
        
        current_entry = (
            f"Sentence {i + 1}: '{sentence}'\n"
            f"Target word: '{lemma}'\n"
            f"Synset Options:\n{synset_descriptions}\n\n"
        )
        
        current_prompt += current_entry
        current_lemma_list.append(lemma)
        
        if (i + 1) % words_per_prompt == 0 or i + 1 == total_length:
            current_prompt += "Please respond only with the synset code for each sentence, in order in the form of a comma-separated list, with no space between commas."
            prompts.append(current_prompt)
            
            lemmas_in_prompt.append(current_lemma_list)
            current_lemma_list = []
            current_prompt = "For each sentence below, choose the correct WordNet synset code for the target word in context.\n\n"

    return prompts, lemmas_in_prompt

def estimate_tokens(prompt, tokenizer):
    response = tokenizer.count_tokens(prompt)
    num_tokens = response.total_tokens
    return num_tokens


def create_prompt_batch(sentences, lemmas):
    prompt = "For each sentence below, choose the correct WordNet synset code for the target word in context.\n\n"
    for i, (sentence, lemma) in enumerate(zip(sentences, lemmas)):
        synset_info = get_synset_codes(lemma)
        # Add each sentence, lemma, and synset options to the prompt
        synset_descriptions = "\n".join([f"{code}: {definition}" for code, definition in synset_info])
        prompt += (
            f"Sentence {i + 1}: '{sentence}'\n"
            f"Target word: '{lemma}'\n"
            f"Synset Options:\n{synset_descriptions}\n\n"
        )
    prompt += "Please respond only with the synset code for each sentence, in order in the form of a comma-separated list."
    
    return prompt


sentences = test_df['Combined Context']
lemmas = test_df['Lemma']

prompt = create_prompt_batch(sentences, lemmas)
num_tokens = estimate_tokens(prompt, tokenizer)

# Given that gemini allows up to 1 million tokens, we don't need to create batches and can include it all as a single prompt
# For this, we will split up the prompts into chunks of lemmas such that the model can focus better


def llm_WSD(sentences, lemmas, words_per_prompt=20, wait_time=5):
    prompts, lemmas_in_prompt = create_prompt_batches(sentences, lemmas, words_per_prompt=words_per_prompt)
    predictions = []
    total_num_replacements = 0
    
    for j, prompt in enumerate(tqdm(prompts, desc="Processing Prompts")):
        response = model.generate_content(prompt)
        text_response = response.text
        cleaned_response = text_response.replace(" ", "").replace("\n", "")
        response_list = cleaned_response.split(",")
        
        if len(response_list) != len(lemmas_in_prompt[j]):
            # In the case where there is an error and no response is produced, we insert the most likely synset
            for i in range(len(lemmas_in_prompt[j])):
               possible_synsets_sets = get_synset_codes(lemmas_in_prompt[j][i])
               possible_synsets = [code for code, definition in possible_synsets_sets]
               if response_list[i] not in possible_synsets:
                   response_list.insert(i, possible_synsets[0])
                   total_num_replacements += 1
        
        predictions += response_list
        
        # Wait time to avoid rate limits
        time.sleep(wait_time)
    
    return predictions, total_num_replacements


df = test_df

sentences = df['Combined Context']
lemmas = df['Lemma']

predictions, num_replacements = llm_WSD(sentences, lemmas, words_per_prompt=50, wait_time=1)

actual_synsets = [wn.lemma_from_key(row['Sense Key'][0]).synset() for _, row in df.iterrows()]


acc, correct_indices, incorrect_indices = calculate_accuracy(predictions, actual_synsets, get_name=True)

print("--" * 20)
print(f"Accuracy for WSD Using the Google Gemnini 1.5 Model: {acc*100:.2f}%")
print("--" * 20)


correct_samples = random.sample(correct_indices, 1) 
incorrect_samples = random.sample(incorrect_indices, 1)

print("\n----------------------------------------- Correct Prediction Debug -----------------------------------------")
for idx in correct_samples:
    prompt_correct = create_prompt_batch([sentences[idx]], [lemmas[idx]])
    prompt_correct = prompt_correct.replace("Please respond only with the synset code for each sentence, in order in the form of a comma-separated list, with no space between commas.", "")
    prompt_correct += 'Please go through each of the available definitions, and justify why the word sense aligns with the definition or not. Finally, once you have gone through each definition, provide the correct synset of those available.'
    response_correct = model.generate_content(prompt_correct)
    
    print(f"\nInstance ID: {df.iloc[idx]['Instance ID']}")
    print(f"Prompt:\n{prompt_correct}")
    print(f"\nResponse:\n{response_correct.text}")
    print(f"\nActual Synset: {actual_synsets[idx]}")

print("\n----------------------------------------- Incorrect Prediction Debug -----------------------------------------")
for idx in incorrect_samples:
    prompt_incorrect = create_prompt_batch([sentences[idx]], [lemmas[idx]])
    prompt_incorrect = prompt_incorrect.replace("Please respond only with the synset code for each sentence, in order in the form of a comma-separated list, with no space between commas.", "")
    prompt_incorrect += 'Please go through each of the available definitions, and justify why the word sense aligns with the definition or not. Finally, once you have gone through each definition, provide the correct synset of those available.'
    
    response_incorrect = model.generate_content(prompt_incorrect)
    
    print(f"\nInstance ID: {df.iloc[idx]['Instance ID']}")
    print(f"Prompt:\n{prompt_incorrect}")
    print(f"\nResponse:\n{response_incorrect.text}")
    print(f"\nActual Synset: {actual_synsets[idx]}")
