# main.py

import os
import subprocess
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import transformers
import torch_optimizer

# print(transformers.__version__) # Sometimes is not working, idk why

# === 0. Setup ===

# Create dataset directory and download files
os.makedirs('dataset', exist_ok=True)

def run_command(command):
    subprocess.run(command, shell=True, check=True)

run_command('wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O dataset/train.json')
run_command('wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O dataset/dev.json')

# === 1. Load Dataset ===

def load_squad_data(path):
    with open(path, 'rb') as f:
        data_dict = json.load(f)

    context_lst, question_lst, answer_lst = [], [], []
    for group in data_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    context_lst.append(context)
                    question_lst.append(question)
                    answer_lst.append(answer)
    return context_lst, question_lst, answer_lst

train_contexts, train_questions, train_answers = load_squad_data('dataset/train.json')
val_contexts, val_questions, val_answers = load_squad_data('dataset/dev.json')

# === 2. Check Data ===

print("Train size:", len(train_contexts))
print("Val size:", len(val_contexts))
print("Example:")
print("Passage:", train_contexts[0])
print("Question:", train_questions[0])
print("Answer:", train_answers[0])

# Sample size
train_sample_size = 10000
val_sample_size = 2000

train_contexts = train_contexts[:train_sample_size]
train_questions = train_questions[:train_sample_size]
train_answers = train_answers[:train_sample_size]

val_contexts = val_contexts[:val_sample_size]
val_questions = val_questions[:val_sample_size]
val_answers = val_answers[:val_sample_size]

# === 3. Fix Answer Indices ===

def fix_answers(answers, contexts):
    for answer, context in zip(answers, contexts):
        text = answer['text']
        start = answer['answer_start']
        end = start + len(text)

        if context[start:end] == text:
            answer['answer_end'] = end
        elif context[start - 1:end - 1] == text:
            answer['answer_start'] = start - 1
            answer['answer_end'] = end - 1
        elif context[start - 2:end - 2] == text:
            answer['answer_start'] = start - 2
            answer['answer_end'] = end - 2

fix_answers(train_answers, train_contexts)
fix_answers(val_answers, val_contexts)

# === 4. Tokenize ===

from transformers import AutoTokenizer, BertForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# print(tokenizer.is_fast)  # True for char_to_token
# from transformers import BertTokenizerFast, BertForQuestionAnswering
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", use_fast=True)


train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    count = 0

    for i in range(len(answers)):
        start = encodings.char_to_token(i, answers[i]['answer_start'], sequence_index=0)
        end = encodings.char_to_token(i, answers[i]['answer_end'], sequence_index=0)

        if start is None:
            start = tokenizer.model_max_length
        if end is None:
            end = encodings.char_to_token(i, answers[i]['answer_end'] - 1, sequence_index=0)
            if end is None:
                count += 1
                end = tokenizer.model_max_length

        start_positions.append(start)
        end_positions.append(end)

    print("Answers skipped due to truncation:", count)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

# === 5. Dataset and DataLoader ===

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

# === 6. Train Model ===

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

import torch_optimizer as optim_lookahead
from torch.optim import AdamW
from tqdm import tqdm

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
adamw = AdamW(model.parameters(), lr=5e-5)
optim = optim_lookahead.Lookahead(adamw)

epochs = 4
train_losses, val_losses = [], []
start_time = time.time()

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            val_loss += outputs[0].item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

print("Total Time:", time.time() - start_time)

# === 7. Plot Loss Curve ===

os.makedirs("output", exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("output/loss_curve.png")
print("Saved loss curve at output/loss_curve.png")
