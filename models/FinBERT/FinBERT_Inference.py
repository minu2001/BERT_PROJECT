import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.path.dirname(__file__), "custom_model_FinBERT")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

def predict_sentiment(texts: list[str]) -> list[int]:
    inputs = tokenizer(texts, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=8)

    all_preds = []
    for batch in tqdm(dataloader, desc="FinBERT Inference"):
        batch_ids, batch_mask = [b.to(device) for b in batch]
        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds
