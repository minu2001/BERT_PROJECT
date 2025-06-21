# FinBERT-Finetune-GPU.py

import torch
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score

data_path = os.path.join(os.path.dirname(__file__), "C:/Users/0120c/BERT_PROJECT/data_file/finance_data.csv")
save_path = os.path.join(os.path.dirname(__file__), "custom_model_FinBERT")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
logging.set_verbosity_error()

df = pd.read_csv(data_path, encoding="cp949", encoding_errors="ignore")
df = df[df['labels'].isin(['negative', 'positive', 'neutral'])]
label_map = {'negative': 0, 'positive': 1, 'neutral': 2}
labels = df['labels'].map(label_map).astype(int).values
data_X = df['sentence'].astype(str).tolist()

model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretra ined(model_name)
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", return_tensors='pt')
input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']

train_ids, val_ids, train_y, val_y, train_mask, val_mask = train_test_split(
    input_ids, labels, attention_mask, test_size=0.2, random_state=2025
)

batch_size = 8
train_dataset = TensorDataset(train_ids, train_mask, torch.tensor(train_y))
val_dataset = TensorDataset(val_ids, val_mask, torch.tensor(val_y))
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * 6)

for epoch in range(8):
    model.train()
    total_loss, train_preds, train_true = 0, [], []

    for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
        batch_ids, batch_mask, batch_labels = [b.to(device) for b in batch]
        model.zero_grad()
        output = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = output.loss
        logits = output.logits
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

        train_preds.extend(preds)
        train_true.extend(batch_labels.cpu().numpy())
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_acc = accuracy_score(train_true, train_preds)
    val_preds, val_true = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"[Eval] Epoch {epoch+1}"):
            batch_ids, batch_mask, batch_labels = [b.to(device) for b in batch]
            output = model(batch_ids, attention_mask=batch_mask)
            preds = torch.argmax(output.logits, dim=1).detach().cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(batch_labels.cpu().numpy())

    val_acc = accuracy_score(val_true, val_preds)
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("✅ 모델 저장 완료:", save_path)
