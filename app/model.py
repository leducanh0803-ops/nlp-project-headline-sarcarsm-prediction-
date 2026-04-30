import torch
from transformers import AutoTokenizer
from src.model import SarcasmModel
from src.config import MODEL_NAME, MAX_LEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = SarcasmModel(MODEL_NAME)
model.load_state_dict(torch.load("weights/model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(text):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        prob = torch.sigmoid(logits).item()

    return prob