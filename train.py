import torch
import torch.nn as nn
from models.transformer import Transformer

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
learning_rate = 3e-4 # Or Noam Scheduler
batch_size = 32

model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        src, trg = batch.src.to(device), batch.trg.to(device)
        
        output = model(src, trg[:, :-1]) # Target shifted for teacher forcing
        output = output.reshape(-1, output.shape[2])
        trg = trg[:, 1:].reshape(-1)
        
        optimizer.zero_grad()
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
