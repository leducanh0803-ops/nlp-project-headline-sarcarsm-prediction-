import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os

from config import *
from dataset import SarcasmDataset
from model import SarcasmModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
data1 = pd.read_json("./data/Sarcasm_Headlines_Dataset.json", lines=True)
data2 = pd.read_json("./data/Sarcasm_Headlines_Dataset_v2.json", lines=True)
data = pd.concat([data1[['is_sarcastic','headline']], data2], axis=0)

X = data['headline']
y = data['is_sarcastic']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = SarcasmDataset(X_train, y_train, tokenizer, MAX_LEN)
val_ds   = SarcasmDataset(X_val, y_val, tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = SarcasmModel(MODEL_NAME).to(DEVICE)

# freeze BERT (optional)
for p in model.bert.parameters():
    p.requires_grad = False

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        x = batch["input_ids"].to(DEVICE)
        m = batch["attention_mask"].to(DEVICE)
        yb = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        out = model(x, m)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

# save
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/model.pt")
print("✅ saved")