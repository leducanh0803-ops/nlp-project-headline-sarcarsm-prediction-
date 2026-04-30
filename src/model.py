from transformers import AutoModel
import torch.nn as nn

class SarcasmModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)  # ✅ FIX
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        return self.fc(x).squeeze(1)