import math
import time
import torch

import torch.nn as nn
from tqdm import tqdm

from data import get_iterators
from baseline import build_model


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          device: torch.device,
          clip: float):
    model.train()
    epoch_loss = 0

    for _, (src, trg) in tqdm(enumerate(iterator)):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        # forward
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device: torch.device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)



def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(batch_size:int, epochs:int, model_path: str=None):
    
    train_iter, valid_iter, test_iter, opts  = get_iterators(batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    INPUT_DIM = opts["src_vocab_size"]
    OUTPUT_DIM = opts["tgt_vocab_size"]
    # ENC_EMB_DIM = 32
    # DEC_EMB_DIM = 32
    # ENC_HID_DIM = 64
    # DEC_HID_DIM = 64
    # ATTN_DIM = 8
    # 3,244,510 parameters
    ENC_EMB_DIM = 12
    DEC_EMB_DIM = 12
    ENC_HID_DIM = 24
    DEC_HID_DIM = 24
    ATTN_DIM = 2
    # 1,202,392
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    CLIP = 1

    model = build_model(INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM,
                ENC_HID_DIM, ENC_DROPOUT, DEC_EMB_DIM,
                DEC_HID_DIM, DEC_DROPOUT, ATTN_DIM, device)
    print(f"The model has {count_parameters(model):,} trainable parameters")

    if model_path is not None:
        print("trying to load model...")
        model.load_state_dict(torch.load(model_path))
        print("model loaded 🎉")
    else:
        print("Initializing weights...")
        model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=opts['TGT_PAD_IDX'])

    best_valid_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, device, CLIP)
        # valid_loss = evaluate(model, valid_iter, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        # print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")

    # save the model
    print("...saving the model...")
    torch.save(model.state_dict(), model_path)
    print("...model saved ✅")

    # test_loss = evaluate(model, test_iter, criterion)
    # print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')