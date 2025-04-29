import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch
import os
from nltk.tokenize import word_tokenize
import nltk
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------
# 1. Define Project Scope
# -------------------
# Goal: Fine-tune GPT-2 to generate creative poetry in the style of classic and modern poems.
# Dataset: Hugging Face 'poem_sentiment' dataset, which contains poem verses.

# -------------------
# 2. Load and Explore Dataset (EDA)
# -------------------
def load_poetry_dataset():
    # Load dataset from Hugging Face
    dataset = load_dataset('poem_sentiment')
    poems = dataset['train']['verse_text']
    return poems

def exploratory_data_analysis(poems, output_dir='eda_plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame({'poem': poems})
    
    # Basic statistics
    df['length'] = df['poem'].apply(len)
    df['word_count'] = df['poem'].apply(lambda x: len(word_tokenize(x)))
    
    print("Dataset Statistics:")
    print(df.describe())
    
    # Plot length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['length'], bins=30, kde=True)
    plt.title('Distribution of Poem Lengths (Characters)')
    plt.xlabel('Length (Characters)')
    plt.savefig(os.path.join(output_dir, 'length_distribution.png'))
    plt.close()
    
    # Plot word count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=30, kde=True)
    plt.title('Distribution of Poem Word Counts')
    plt.xlabel('Word Count')
    plt.savefig(os.path.join(output_dir, 'word_count_distribution.png'))
    plt.close()
    
    # Common words
    all_text = ' '.join(df['poem'])
    words = word_tokenize(all_text.lower())
    word_freq = pd.Series(words).value_counts()[:20]
    
    plt.figure(figsize=(12, 6))
    word_freq.plot(kind='bar')
    plt.title('Top 20 Most Common Words')
    plt.savefig(os.path.join(output_dir, 'common_words.png'))
    plt.close()

# -------------------
# 3. Preprocess Dataset
# -------------------
def preprocess_poems(poems):
    # Clean text: remove special characters, normalize spaces
    cleaned_poems = [re.sub(r'\s+', ' ', poem.strip()) for poem in poems]
    return cleaned_poems

def prepare_dataset(poems, tokenizer, max_length=128):
    # Tokenize poems
    encodings = tokenizer(
        poems,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create dataset for training
    class PoetryDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings['input_ids'])
        
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item
    
    return PoetryDataset(encodings)

# -------------------
# 4. Fine-Tune GPT-2 Model
# -------------------
def fine_tune_model(dataset, output_dir='fine_tuned_gpt2', hyperparams=None):
    # Load model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Default hyperparameters
    default_hyperparams = {
        'num_train_epochs': 3,
        'per_device_train_batch_size': 8,
        'learning_rate': 5e-5,
        'warmup_steps': 100,
    }
    
    # Update with provided hyperparameters
    if hyperparams:
        default_hyperparams.update(hyperparams)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=default_hyperparams['num_train_epochs'],
        per_device_train_batch_size=default_hyperparams['per_device_train_batch_size'],
        learning_rate=default_hyperparams['learning_rate'],
        warmup_steps=default_hyperparams['warmup_steps'],
        save_strategy='epoch',
        logging_strategy='epoch',
        fp16=torch.cuda.is_available(),
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

# -------------------
# 5. Generate Text
# -------------------
def generate_poem(model, tokenizer, prompt="The moonlight glows", max_length=100, num_return_sequences=3):
    device = torch.device("cpu")
    model.to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    poems = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return poems

# -------------------
# 6. Qualitative Analysis
# -------------------
def analyze_generated_poems(poems):
    analysis = {
        'creativity': [],
        'coherence': [],
        'relevance': []
    }
    
    for i, poem in enumerate(poems):
        print(f"\nGenerated Poem {i+1}:\n{poem}")
        
        # Qualitative scoring (1-5 scale, manual for simplicity)
        creativity = 3  # Adjust based on novelty (e.g., unique metaphors)
        coherence = 4   # Adjust based on logical flow
        relevance = 4   # Adjust based on alignment with poetry style
        
        if len(set(poem.split())) / len(poem.split()) < 0.5:
            creativity -= 1  # Penalize repetition
        if re.search(r'\b\w+\b.*\b\w+\b.*\b\w+\b', poem):
            coherence += 1  # Reward structured sentences
        if 'moon' in poem.lower() or 'light' in poem.lower():
            relevance += 1  # Reward thematic consistency
        
        analysis['creativity'].append(min(max(creativity, 1), 5))
        analysis['coherence'].append(min(max(coherence, 1), 5))
        analysis['relevance'].append(min(max(relevance, 1), 5))
    
    # Summarize
    print("\nQualitative Analysis:")
    for metric in analysis:
        avg_score = np.mean(analysis[metric])
        print(f"{metric.capitalize()}: Average Score = {avg_score:.2f}/5")
    
    return analysis

# -------------------
# 7. Main Execution
# -------------------
def main():
    # Load and explore data
    poems = load_poetry_dataset()
    exploratory_data_analysis(poems)
    
    # Preprocess and prepare dataset
    cleaned_poems = preprocess_poems(poems)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = prepare_dataset(cleaned_poems, tokenizer)
    
    # Experiment with two hyperparameter sets
    hyperparams_list = [
        {'num_train_epochs': 3, 'learning_rate': 5e-5, 'per_device_train_batch_size': 8},
        {'num_train_epochs': 4, 'learning_rate': 3e-5, 'per_device_train_batch_size': 4},
    ]
    
    for i, hyperparams in enumerate(hyperparams_list):
        print(f"\nTraining with Hyperparameters Set {i+1}: {hyperparams}")
        output_dir = f'fine_tuned_gpt2_set_{i+1}'
        model, tokenizer = fine_tune_model(dataset, output_dir, hyperparams)
        
        # Generate and analyze poems
        generated_poems = generate_poem(model, tokenizer)
        analyze_generated_poems(generated_poems)

# -------------------
# 8. Instructions for Replication
# -------------------
"""
Instructions for Replication:
1. Install dependencies:
   ```bash
   pip install transformers datasets torch pandas numpy matplotlib seaborn nltk
   ```
2. Ensure CUDA is available for GPU acceleration (optional but recommended).
3. Save this script as `poetry_generator.py`.
4. Run the script:
   ```bash
   python poetry_generator.py
   ```
5. Outputs:
   - EDA plots in `eda_plots/` directory.
   - Fine-tuned models in `fine_tuned_gpt2_set_1/` and `fine_tuned_gpt2_set_2/`.
   - Generated poems and qualitative analysis printed to console.
6. Notes:
   - The `datasets` library is required to load the poem dataset.
   - Dataset is automatically downloaded via Hugging Face.
   - Adjust `max_length` or hyperparameters for different results.
   - Use a GPU for faster training.
   - If the error `ModuleNotFoundError: No module named 'datasets'` occurs, ensure `datasets` is installed using `pip install datasets`.
"""

if __name__ == '__main__':
    main()