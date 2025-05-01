import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
from config import DEVICE, DATA_FILE, CHECKPOINT_DIR, load_and_clean_data, VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT
from model.transformer import StoryTellerTransformer
from dataloader.tokenizer_utils import load_tokenizer
from config import TOKENIZER_PATH

class SimCSEProjectionHead(nn.Module):
    def __init__(self, d_model, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
    
    def forward(self, x):
        return self.mlp(x)

def simcse_loss(emb1, emb2, temperature=0.05):
    batch_size = emb1.size(0)
    labels = torch.arange(batch_size).to(emb1.device)
    similarity = torch.matmul(emb1, emb2.T) / temperature
    loss = nn.CrossEntropyLoss()(similarity, labels)
    return loss

def pretrain_simcse(model, tokenizer, data, epochs=3, batch_size=32):
    model.train()
    projection_head = SimCSEProjectionHead(model.d_model).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection_head.parameters()),
        lr=1e-5
    )
    
    dataset = [item["text"] for item in data if item["text"].strip()]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            encoded = [tokenizer.encode(text, out_type=int) for text in batch]
            max_len = min(max(len(seq) for seq in encoded), 256)
            input_ids = torch.full((len(encoded), max_len), 0, dtype=torch.long, device=DEVICE)
            for i, seq in enumerate(encoded):
                input_ids[i, :min(len(seq), max_len)] = torch.tensor(seq[:max_len], dtype=torch.long, device=DEVICE)
            
            emb1 = model(input_ids, apply_dropout=True).mean(dim=1)
            emb2 = model(input_ids, apply_dropout=True).mean(dim=1)
            emb1 = projection_head(emb1)
            emb2 = projection_head(emb2)
            
            loss = simcse_loss(emb1, emb2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        logging.info(f"SimCSE Pre-training Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        print(f"✅ SimCSE Pre-training Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    pretrained_path = os.path.join(CHECKPOINT_DIR, "pretrained_transformer.pth")
    try:
        torch.save(model.state_dict(), pretrained_path)
        logging.info(f"Saved pre-trained model: {pretrained_path}")
        print(f"✅ Saved pre-trained model: {pretrained_path}")
    except Exception as e:
        logging.error(f"Failed to save pre-trained model: {e}")
        print(f"❌ Pre-trained model save error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    try:
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        print(f"✅ Loaded tokenizer from {TOKENIZER_PATH}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load tokenizer: {e}")
    
    model = StoryTellerTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)
    
    try:
        data = load_and_clean_data(DATA_FILE)
        print(f"✅ Loaded {len(data)} samples for pre-training")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load data: {e}")
    
    pretrain_simcse(model, tokenizer, data)