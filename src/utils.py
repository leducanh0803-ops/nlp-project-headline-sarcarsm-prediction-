import torch

def accuracy_from_probs(y_pred, y_true, threshold=0.5):
    preds = (y_pred > threshold).float()
    return (preds == y_true).float().mean().item()

def move_batch_to_device(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "label": batch["label"].to(device),
    }